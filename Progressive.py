import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.interpolate import interp1d
import struct
import warnings
warnings.filterwarnings('ignore')


class RingDataParser:
    """Parse hex data from smart ring accelerometer."""
    
    def parse_hex_packet(self, hex_str):
        """Parse a single hex packet from the ring.
        
        Expected format: a1 XX ... (various packet types)
        """
        if not hex_str or len(hex_str) < 4:
            return None
        
        # Remove any spaces and convert to bytes
        hex_clean = hex_str.replace(' ', '')
        try:
            data = bytes.fromhex(hex_clean)
        except:
            return None
        
        if len(data) < 2:
            return None
        
        # Check header
        if data[0] != 0xa1:
            return None
        
        packet_type = data[1]
        
        # Type 03: Accelerometer data (most relevant for us)
        if packet_type == 0x03 and len(data) >= 8:
            # Bytes 2-3: accel_x (signed 16-bit)
            # Bytes 4-5: accel_y (signed 16-bit)  
            # Bytes 6-7: accel_z (signed 16-bit)
            accel_x = struct.unpack('<h', data[2:4])[0]
            accel_y = struct.unpack('<h', data[4:6])[0]
            accel_z = struct.unpack('<h', data[6:8])[0]
            
            return {
                'type': 'accel',
                'accel_x': accel_x,
                'accel_y': accel_y,
                'accel_z': accel_z
            }
        
        return None
    
    def parse_csv_row(self, row):
        """Parse a single CSV row."""
        if len(row) < 3:
            return None
        
        timestamp = row[0]
        hex_data = row[1]
        
        # Try to parse accelerometer data from columns 2-4
        try:
            if row[2] != '' and row[3] != '' and row[4] != '':
                return {
                    'timestamp': timestamp,
                    'type': 'accel',
                    'accel_x': int(row[2]),
                    'accel_y': int(row[3]),
                    'accel_z': int(row[4])
                }
        except:
            pass
        
        # Otherwise try parsing hex
        parsed = self.parse_hex_packet(hex_data)
        if parsed:
            parsed['timestamp'] = timestamp
            return parsed
        
        return None
    
    def load_from_csv_rows(self, csv_rows):
        """Load accelerometer data from list of CSV rows.
        
        Args:
            csv_rows: List of lists, each row is [timestamp, hex, col2, col3, col4, ...]
        
        Returns:
            pd.DataFrame with columns: timestamp, accel_x, accel_y, accel_z
        """
        data = []
        for row in csv_rows:
            parsed = self.parse_csv_row(row)
            if parsed and parsed['type'] == 'accel':
                data.append({
                    'timestamp': parsed['timestamp'],
                    'accel_x': parsed['accel_x'],
                    'accel_y': parsed['accel_y'],
                    'accel_z': parsed['accel_z']
                })
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df


class MicrographiaAnalyzer:
    """Analyze handwriting for progressive micrographia (shrinking writing)."""
    
    def __init__(self, sample_rate=50):
        self.sample_rate = sample_rate
    
    def compute_magnitude(self, df):
        """Compute acceleration magnitude."""
        return np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)
    
    def segment_strokes(self, accel_mag, threshold_percentile=30):
        """Segment continuous writing into individual strokes.
        
        A stroke is a period of high acceleration (active writing).
        """
        # Threshold for detecting active writing
        threshold = np.percentile(accel_mag, threshold_percentile)
        
        # Find periods above threshold
        active = accel_mag > threshold
        
        # Find stroke boundaries
        strokes = []
        in_stroke = False
        start_idx = 0
        
        for i, is_active in enumerate(active):
            if is_active and not in_stroke:
                start_idx = i
                in_stroke = True
            elif not is_active and in_stroke:
                if i - start_idx > 10:  # Minimum stroke length
                    strokes.append((start_idx, i))
                in_stroke = False
        
        # Handle last stroke
        if in_stroke and len(accel_mag) - start_idx > 10:
            strokes.append((start_idx, len(accel_mag)))
        
        return strokes
    
    def compute_stroke_amplitude(self, accel_mag, stroke_bounds):
        """Compute amplitude (size proxy) for each stroke."""
        amplitudes = []
        
        for start, end in stroke_bounds:
            stroke_data = accel_mag[start:end]
            
            # Multiple amplitude metrics
            amp_features = {
                'peak': np.max(stroke_data),
                'mean': np.mean(stroke_data),
                'range': np.max(stroke_data) - np.min(stroke_data),
                'std': np.std(stroke_data),
                'rms': np.sqrt(np.mean(stroke_data**2)),
                'start_idx': start,
                'end_idx': end,
                'duration': end - start
            }
            amplitudes.append(amp_features)
        
        return amplitudes
    
    def detect_progressive_decline(self, stroke_amplitudes, metric='rms'):
        """Detect if writing size progressively decreases.
        
        Returns:
            dict with decline detection results
        """
        if len(stroke_amplitudes) < 3:
            return {
                'has_micrographia': False,
                'confidence': 0.0,
                'reason': 'insufficient_strokes'
            }
        
        # Extract amplitude metric over time
        amp_values = np.array([s[metric] for s in stroke_amplitudes])
        positions = np.arange(len(amp_values))
        
        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(positions, amp_values)
        
        # Compute decline percentage
        initial_amp = amp_values[:3].mean()
        final_amp = amp_values[-3:].mean()
        decline_pct = ((initial_amp - final_amp) / initial_amp) * 100 if initial_amp > 0 else 0
        
        # Criteria for progressive micrographia:
        # 1. Significant negative slope (p < 0.05)
        # 2. R² > 0.3 (moderate fit)
        # 3. Decline > 15%
        
        r_squared = r_value ** 2
        is_significant = p_value < 0.05
        is_declining = slope < 0
        is_moderate_fit = r_squared > 0.3
        is_substantial_decline = decline_pct > 15
        
        # Calculate confidence score (0-1)
        confidence = 0.0
        if is_declining and is_significant:
            confidence += 0.4
        if is_moderate_fit:
            confidence += 0.3 * min(r_squared / 0.7, 1.0)  # Scale to 0.7 max
        if is_substantial_decline:
            confidence += 0.3 * min(decline_pct / 40, 1.0)  # Scale to 40% decline
        
        has_micrographia = (is_significant and is_declining and 
                           is_moderate_fit and is_substantial_decline)
        
        return {
            'has_micrographia': has_micrographia,
            'confidence': confidence,
            'slope': slope,
            'r_squared': r_squared,
            'p_value': p_value,
            'decline_percentage': decline_pct,
            'initial_amplitude': initial_amp,
            'final_amplitude': final_amp,
            'n_strokes': len(amp_values),
            'amplitude_trend': amp_values.tolist()
        }
    
    def analyze_writing_sample(self, df):
        """Full analysis pipeline for a writing sample.
        
        Args:
            df: DataFrame with accel_x, accel_y, accel_z columns
        
        Returns:
            dict with analysis results
        """
        if df is None or len(df) < 50:
            return {
                'error': 'insufficient_data',
                'has_micrographia': False,
                'confidence': 0.0
            }
        
        # Compute magnitude
        accel_mag = self.compute_magnitude(df)
        
        # Segment into strokes
        strokes = self.segment_strokes(accel_mag)
        
        if len(strokes) < 3:
            return {
                'error': 'insufficient_strokes',
                'has_micrographia': False,
                'confidence': 0.0,
                'n_strokes': len(strokes)
            }
        
        # Compute stroke amplitudes
        stroke_amps = self.compute_stroke_amplitude(accel_mag, strokes)
        
        # Detect progressive decline
        results = self.detect_progressive_decline(stroke_amps, metric='rms')
        
        # Add additional context
        results['total_samples'] = len(df)
        results['duration_seconds'] = (df['timestamp'].iloc[-1] - 
                                      df['timestamp'].iloc[0]).total_seconds()
        
        return results


class ParkinsonsMicrographiaDetector:
    """High-level detector for Parkinson's based on micrographia."""
    
    def __init__(self):
        self.parser = RingDataParser()
        self.analyzer = MicrographiaAnalyzer()
    
    def predict_from_csv_rows(self, csv_rows):
        """Main prediction function.
        
        Args:
            csv_rows: List of CSV rows from ring data
        
        Returns:
            dict with prediction and details
        """
        # Parse data
        df = self.parser.load_from_csv_rows(csv_rows)
        
        if df is None:
            return {
                'has_parkinsons': False,
                'confidence': 0.0,
                'error': 'no_valid_accelerometer_data'
            }
        
        # Analyze for micrographia
        results = self.analyzer.analyze_writing_sample(df)
        
        # Map to Parkinson's prediction
        has_parkinsons = results.get('has_micrographia', False)
        confidence = results.get('confidence', 0.0)
        
        # Enhance confidence based on additional factors
        if 'decline_percentage' in results and results['decline_percentage'] > 30:
            confidence = min(confidence + 0.1, 1.0)
        
        prediction = {
            'has_parkinsons': has_parkinsons,
            'confidence': confidence,
            'risk_level': self._get_risk_level(confidence),
            'micrographia_details': results
        }
        
        return prediction
    
    def _get_risk_level(self, confidence):
        """Convert confidence to risk level."""
        if confidence >= 0.7:
            return 'HIGH'
        elif confidence >= 0.4:
            return 'MODERATE'
        elif confidence >= 0.2:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def predict_from_file(self, csv_file):
        """Predict from CSV file."""
        df_raw = pd.read_csv(csv_file, header=None)
        csv_rows = df_raw.values.tolist()
        return self.predict_from_csv_rows(csv_rows)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Progressive Micrographia Detector for Parkinson's Disease")
    print("=" * 70)
    
    # Example data from your format
    example_data = [
        ['2025-09-21T16:35:23.926910', 'a103011a15af120d00000000000000a2', 301, 26, 351, '', '', '', '', '', '', '', ''],
        ['2025-09-21T16:35:24.945971', 'a1011a731807131a04ed01000000006d', '', '', '', '', '', '', '', 6771, 7, 26, 237],
        ['2025-09-21T16:35:24.947189', 'a102000111eb000105f900000000009f', '', '', '', 1, 4587, 1, 1529, '', '', '', ''],
        ['2025-09-21T16:35:24.947189', 'a103021d182411a200000000000000b2', 274, 45, -1660, '', '', '', '', '', '', '', ''],
        ['2025-09-21T16:35:25.965635', 'a1011a811807131a04ed01000000007b', '', '', '', '', '', '', '', 6785, 7, 26, 237],
        ['2025-09-21T16:35:25.966842', 'a1020e4b11eb000105f90000000000f7', '', '', '', 3659, 4587, 1, 1529, '', '', '', ''],
        ['2025-09-21T16:35:25.966842', 'a103017416880feb00000000000000b1', -1797, 20, 360, '', '', '', '', '', '', '', ''],
        ['2025-09-21T16:35:26.925514', 'a1011a7a1807131a04ed010000000074', '', '', '', '', '', '', '', 6778, 7, 26, 237],
        ['2025-09-21T16:35:26.925514', 'a1020d9a11eb000105f9000000000045', '', '', '', 3482, 4587, 1, 1529, '', '', '', ''],
        ['2025-09-21T16:35:26.926877', 'a10301d715150f2800000000000000dd', -1800, 23, 341, '', '', '', '', '', '', '', ''],
        ['2025-09-21T16:35:27.945472', 'a1011a6f1807131a04ed010000000069', '', '', '', '', '', '', '', 6767, 7, 26, 237],
        ['2025-09-21T16:35:27.946473', 'a1020d2611eb000105f90000000000d1', '', '', '', 3366, 4587, 1, 1529, '', '', '', ''],
        ['2025-09-21T16:35:27.946473', 'a103010618f3122a00000000000000f2', 298, 22, -1661, '', '', '', '', '', '', '', ''],
        ['2025-09-21T16:35:28.905183', 'a1011a721807131a04ed01000000006c', '', '', '', '', '', '', '', 6770, 7, 26, 237],
        ['2025-09-21T16:35:28.965458', 'a1020c9e11eb000105f9000000000048', '', '', '', 3230, 4587, 1, 1529, '', '', '', ''],
        ['2025-09-21T16:35:28.965458', 'a103032914f60eda00000000000000c2', -1814, 57, 326, '', '', '', '', '', '', '', ''],
        ['2025-09-21T16:35:29.925458', 'a1011a721807131a04ed01000000006c', '', '', '', '', '', '', '', 6770, 7, 26, 237],
        ['2025-09-21T16:35:29.926460', 'a1020c8711eb000105f9000000000031', '', '', '', 3207, 4587, 1, 1529, '', '', '', '']
    ]
    
    # Initialize detector
    detector = ParkinsonsMicrographiaDetector()
    
    # Make prediction
    print("\nAnalyzing writing sample...")
    result = detector.predict_from_csv_rows(example_data)
    
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    print(f"\nHas Parkinson's (Micrographia): {result['has_parkinsons']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
    
    if 'error' in result:
        print(f"\nError: {result['error']}")
    
    details = result.get('micrographia_details', {})
    if 'n_strokes' in details:
        print(f"\nNumber of strokes detected: {details['n_strokes']}")
        print(f"Decline percentage: {details.get('decline_percentage', 0):.1f}%")
        print(f"Statistical significance (p-value): {details.get('p_value', 1.0):.4f}")
        print(f"Fit quality (R²): {details.get('r_squared', 0):.3f}")
        
        if 'initial_amplitude' in details:
            print(f"\nAmplitude change:")
            print(f"  Initial: {details['initial_amplitude']:.1f}")
            print(f"  Final: {details['final_amplitude']:.1f}")
    
    # Show interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if result['has_parkinsons']:
        print("\n⚠️  Progressive micrographia detected!")
        print("The handwriting shows statistically significant decline in size")
        print("over the course of the sentence, which is a hallmark of Parkinson's.")
        print("\nRECOMMENDATION: Consult with a neurologist for professional evaluation.")
    else:
        print("\n✓ No significant progressive micrographia detected.")
        print("Writing amplitude remains relatively stable throughout the sample.")
        
        if result['confidence'] > 0.2:
            print("\nNote: Some decline was observed but did not meet diagnostic criteria.")
            print("Consider collecting more samples or monitoring over time.")
    
    print("\n" + "=" * 70)
    print("\nTo use with your own data:")
    print("  detector = ParkinsonsMicrographiaDetector()")
    print("  result = detector.predict_from_csv_rows(your_csv_rows)")
    print("  # or")
    print("  result = detector.predict_from_file('your_data.csv')")
