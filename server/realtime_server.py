"""
Real-time EMG Classification Server
Allows dynamic file selection and continuous prediction streaming.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import threading
import time
from collections import deque
import torch
import pandas as pd
import numpy as np

from NN.predict import load_trained_model
from DATA.Data_Conversion import MOVEMENT_LABELS, SEVERITY_LABELS


class EMGRealtimeServer:
    """
    Server that simulates real-time EMG prediction by streaming data from CSV files.
    Allows dynamic file switching via user commands.
    """
    
    def __init__(self, model_path="./models/final_model_full.pth", window_size=100):
        """
        Initialize the real-time server.
        
        Args:
            model_path: Path to trained model
            window_size: Number of samples for prediction window
        """
        print("="*60)
        print("EMG Real-Time Prediction Server")
        print("="*60)
        
        # Load model
        print(f"\nLoading model from {model_path}...")
        self.model, checkpoint = load_trained_model(model_path)
        self.device = next(self.model.parameters()).device
        self.window_size = window_size
        
        print(f"Model loaded successfully!")
        print(f"Movement Accuracy: {checkpoint['test_metrics']['movement_acc']*100:.2f}%")
        print(f"Severity Accuracy: {checkpoint['test_metrics']['severity_acc']*100:.2f}%")
        
        # Streaming state
        self.current_file = None
        self.emg_data = None
        self.current_index = 0
        self.buffer = deque(maxlen=window_size)
        self.is_streaming = False
        self.stream_thread = None
        
        # Prediction settings
        self.prediction_rate = 10  # Hz (predictions per second)
        self.samples_per_prediction = 5  # New samples to read before next prediction
        
        # Latest prediction
        self.latest_prediction = None
        self.prediction_lock = threading.Lock()
        
    def load_file(self, csv_path):
        """
        Load a new CSV file for streaming.
        
        Args:
            csv_path: Path to CSV file
        """
        if not os.path.exists(csv_path):
            print(f"Error: File not found - {csv_path}")
            return False
        
        try:
            # Load data
            df = pd.read_csv(csv_path)
            self.emg_data = df.values  # numpy array [samples, 8]
            self.current_file = csv_path
            self.current_index = 0
            self.buffer.clear()
            
            print(f"\n[LOADED] {os.path.basename(csv_path)}")
            print(f"  Samples: {self.emg_data.shape[0]}")
            print(f"  Channels: {self.emg_data.shape[1]}")
            print(f"  Duration: ~{self.emg_data.shape[0]/200:.2f}s (at 200 Hz)")
            
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def start_streaming(self):
        """Start streaming predictions from current file."""
        if self.current_file is None:
            print("Error: No file loaded. Use 'load <filename>' first.")
            return
        
        if self.is_streaming:
            print("Already streaming!")
            return
        
        self.is_streaming = True
        self.current_index = 0
        self.buffer.clear()
        
        # Start streaming thread
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()
        
        print(f"\n[STREAMING] Started real-time prediction from {os.path.basename(self.current_file)}")
        print(f"  Update rate: {self.prediction_rate} Hz")
        print(f"  Window size: {self.window_size} samples")
    
    def stop_streaming(self):
        """Stop streaming predictions."""
        if not self.is_streaming:
            print("Not currently streaming.")
            return
        
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
        
        print("\n[STOPPED] Streaming halted.")
    
    def _stream_loop(self):
        """Internal streaming loop (runs in separate thread)."""
        # Collect all predictions across entire file
        all_movement_probs = []
        all_severity_probs = []
        
        print("\nProcessing file...")
        
        while self.is_streaming and self.current_index < len(self.emg_data):
            # Read next batch of samples
            end_idx = min(self.current_index + self.samples_per_prediction, len(self.emg_data))
            new_samples = self.emg_data[self.current_index:end_idx]
            
            # Add to buffer
            for sample in new_samples:
                self.buffer.append(sample)
            
            self.current_index = end_idx
            
            # Make prediction if buffer is full
            if len(self.buffer) == self.window_size:
                prediction_probs = self._get_prediction_probs()
                all_movement_probs.append(prediction_probs['movement_probs'])
                all_severity_probs.append(prediction_probs['severity_probs'])
                
                # Update progress (every 10%)
                progress = (self.current_index / len(self.emg_data)) * 100
                if int(progress) % 10 == 0 and len(all_movement_probs) > 1:
                    prev_progress = ((self.current_index - self.samples_per_prediction) / len(self.emg_data)) * 100
                    if int(prev_progress) % 10 != int(progress) % 10:
                        print(f"  Progress: {int(progress)}%")
            
            # Sleep to maintain realistic timing
            time.sleep(1.0 / self.prediction_rate)
        
        # Aggregate all predictions
        if all_movement_probs and self.is_streaming:
            avg_movement_probs = np.mean(all_movement_probs, axis=0)
            avg_severity_probs = np.mean(all_severity_probs, axis=0)
            
            movement_pred = np.argmax(avg_movement_probs)
            severity_pred = np.argmax(avg_severity_probs)
            
            final_prediction = {
                'movement': MOVEMENT_LABELS[movement_pred],
                'movement_conf': avg_movement_probs[movement_pred],
                'severity': SEVERITY_LABELS[severity_pred],
                'severity_conf': avg_severity_probs[severity_pred],
                'num_windows': len(all_movement_probs)
            }
            
            with self.prediction_lock:
                self.latest_prediction = final_prediction
            
            # Print final result
            print("\n" + "="*60)
            print("FINAL PREDICTION")
            print("="*60)
            print(f"  Movement: {final_prediction['movement']:20s} ({final_prediction['movement_conf']*100:5.1f}%)")
            print(f"  Severity: {final_prediction['severity']:10s} ({final_prediction['severity_conf']*100:5.1f}%)")
            print(f"  Windows analyzed: {final_prediction['num_windows']}")
            print("="*60)
        
        # End of file
        if self.is_streaming:
            self.is_streaming = False
    
    def _get_prediction_probs(self):
        """Get probability distributions from current buffer."""
        # Convert buffer to tensor
        buffer_array = np.array(list(self.buffer))
        emg_tensor = torch.tensor(buffer_array, dtype=torch.float32).unsqueeze(0)  # [1, window_size, 8]
        emg_tensor = emg_tensor.to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            movement_logits, severity_logits = self.model(emg_tensor)
            
            movement_probs = torch.softmax(movement_logits, dim=1)
            severity_probs = torch.softmax(severity_logits, dim=1)
        
        return {
            'movement_probs': movement_probs[0].cpu().numpy(),
            'severity_probs': severity_probs[0].cpu().numpy()
        }
    
    def get_latest_prediction(self):
        """Get the most recent prediction (thread-safe)."""
        with self.prediction_lock:
            return self.latest_prediction
    
    def list_available_files(self, directory="./DATA/Example_data"):
        """List available CSV files for loading."""
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            return []
        
        csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        csv_files.sort()
        
        # Parse filenames to show movement and severity
        movement_map = {
            'C1': 'No Movement',
            'C2': 'Wrist Flexion',
            'C3': 'Wrist Extension',
            'C4': 'Wrist Pronation',
            'C5': 'Wrist Supination',
            'C6': 'Chuck Grip',
            'C7': 'Hand Open'
        }
        
        print(f"\n{'#':<4} {'Movement':<20} {'Severity':<10}")
        print("-" * 60)
        for i, filename in enumerate(csv_files, 1):
            # Parse: S1_Hard_C7_R1.csv -> Hard, C7
            parts = filename.replace('.csv', '').split('_')
            severity = parts[1] if len(parts) > 1 else '?'
            movement_code = parts[2] if len(parts) > 2 else '?'
            movement_name = movement_map.get(movement_code, movement_code)
            
            print(f"{i:<4} {movement_name:<20} {severity:<10}")
        print("-" * 60)
        
        return csv_files
    
    def run_interactive(self):
        """Run interactive command loop."""
        print("\n" + "="*60)
        print("Interactive Mode - Available Commands:")
        print("="*60)
        print("  list                    - List available movements")
        print("  load <number>           - Load file by number")
        print("  start                   - Start prediction")
        print("  stop                    - Stop prediction")
        print("  status                  - Show current status")
        print("  help                    - Show this help")
        print("  quit / exit             - Exit server")
        print("="*60)
        
        available_files = self.list_available_files()
        
        while True:
            try:
                command = input("\nserver> ").strip().lower()
                
                if not command:
                    continue
                
                parts = command.split()
                cmd = parts[0]
                
                if cmd in ['quit', 'exit']:
                    self.stop_streaming()
                    print("Shutting down server...")
                    break
                
                elif cmd == 'help':
                    print("\nAvailable Commands:")
                    print("  list          - List available movements")
                    print("  load <#>      - Load file by number")
                    print("  start         - Start prediction")
                    print("  stop          - Stop prediction")
                    print("  status        - Show current status")
                    print("  quit/exit     - Exit server")
                
                elif cmd == 'list':
                    available_files = self.list_available_files()
                
                elif cmd == 'load':
                    if len(parts) < 2:
                        print("Usage: load <number>")
                        print("Use 'list' to see available files")
                        continue
                    
                    # Stop streaming if active
                    if self.is_streaming:
                        self.stop_streaming()
                        time.sleep(0.5)
                    
                    # Load by number only
                    if parts[1].isdigit():
                        idx = int(parts[1]) - 1
                        if 0 <= idx < len(available_files):
                            filepath = os.path.join("./DATA/Example_data", available_files[idx])
                            self.load_file(filepath)
                        else:
                            print(f"Invalid number. Choose 1-{len(available_files)}")
                    else:
                        print("Please use a number. Use 'list' to see options.")
                
                elif cmd == 'start':
                    self.start_streaming()
                
                elif cmd == 'stop':
                    self.stop_streaming()
                
                elif cmd == 'status':
                    print("\n" + "="*60)
                    print("Server Status")
                    print("="*60)
                    print(f"  Current file: {os.path.basename(self.current_file) if self.current_file else 'None'}")
                    print(f"  Streaming: {'Yes' if self.is_streaming else 'No'}")
                    if self.current_file and self.emg_data is not None:
                        progress = (self.current_index / len(self.emg_data)) * 100
                        print(f"  Progress: {self.current_index}/{len(self.emg_data)} ({progress:.1f}%)")
                    print(f"  Prediction rate: {self.prediction_rate} Hz")
                    print(f"  Window size: {self.window_size} samples")
                    
                    if self.latest_prediction:
                        print(f"\n  Latest prediction:")
                        print(f"    Movement: {self.latest_prediction['movement']} ({self.latest_prediction['movement_conf']*100:.1f}%)")
                        print(f"    Severity: {self.latest_prediction['severity']} ({self.latest_prediction['severity_conf']*100:.1f}%)")
                    print("="*60)
                
                else:
                    print(f"Unknown command: {cmd}. Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit.")
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main entry point."""
    server = EMGRealtimeServer(
        model_path="./models/final_model_full.pth",
        window_size=100
    )
    
    server.run_interactive()


if __name__ == "__main__":
    main()
