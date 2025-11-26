import socket
import re
import os
import sys
import json

import numpy as np

from lwe_constants import N, Q, K_ERROR, a
from ftp_constants import SERVER_HOST, FTP_PORT, BUFFER_SIZE, TIMEOUT

from r_lwe import KeyGen, HelpRec, Rec
from poly_ops import PO # Get the pre-instantiated PolyOps object

class FTPClient:
    """Manages the S-FTP control and data connections."""
    def __init__(self):
        self.control_socket = None
        self.data_ip = None
        self.data_port = None
        self.shared_key = None # The R-LWE Shared Secret K

    def connect(self):
        """Connects the control socket and initiates key exchange."""
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control_socket.settimeout(TIMEOUT)
        try:
            self.control_socket.connect((SERVER_HOST, FTP_PORT))
            print(f"[*] Connected to server at {SERVER_HOST}:{FTP_PORT}")
            
            # Initial 220 Welcome
            initial_response = self._receive_response()
            if not initial_response or not initial_response.startswith('220'):
                raise Exception("Did not receive initial welcome.")
                
            # Perform R-LWE Key Exchange immediately
            if not self._handle_key_exchange():
                return False
                
            return True
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            self.control_socket = None
            return False

    def _send_command(self, command_line):
        """Sends a command string to the server."""
        try:
            full_command = f"{command_line}\r\n"
            self.control_socket.sendall(full_command.encode('utf-8'))
        except Exception as e:
            print(f"[ERROR] Failed to send command: {e}")
            return None

    def _receive_response(self):
        """Receives and returns the server's standard FTP response."""
        try:
            response = self.control_socket.recv(BUFFER_SIZE).decode('utf-8').strip()
            if response:
                print(f"[SERVER]: {response}")
            return response
        except socket.timeout:
            print("[ERROR] Server response timed out.")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to receive response: {e}")
            return None

    def _send_json(self, data):
        """Sends structured data (polys) over the control channel."""
        try:
            json_data = json.dumps(data)
            self.control_socket.sendall(json_data.encode('utf-8') + b'\r\n')
        except Exception as e:
            print(f"[ERROR] Failed to send JSON: {e}")

    def _recv_json(self):
        """Receives structured data (polys) from the control channel."""
        data = b''
        while True:
            chunk = self.control_socket.recv(1)
            if chunk == b'\n':
                break
            if chunk != b'\r':
                data += chunk
        
        try:
            return json.loads(data.decode('utf-8'))
        except json.JSONDecodeError:
            print(f"[ERROR] JSON decoding failed for data: {data}")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to receive JSON: {e}")
            return None

    def _open_data_connection(self):
        """Connects the data socket using the stored IP/Port from the PASV response."""
        if not self.data_ip or not self.data_port:
            print("[ERROR] Data connection parameters missing. Did PASV succeed?")
            return None
        
        data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        data_socket.settimeout(TIMEOUT)
        try:
            data_socket.connect((self.data_ip, self.data_port))
            print(f"[*] Data connection established to {self.data_ip}:{self.data_port}")
            return data_socket
        except Exception as e:
            print(f"[ERROR] Failed to establish data connection: {e}")
            return None

    def xor_data(self, data_bytes):
        """Performs XOR encryption/decryption using the repeating R-LWE key (K)."""
        if self.shared_key is None:
            raise ValueError("Shared key not established.")

        key_stream = np.array(self.shared_key, dtype=np.uint8)
        key_len = len(key_stream)
        
        encrypted_bytes = bytearray(len(data_bytes))
        
        for i in range(len(data_bytes)):
            key_byte = key_stream[i % key_len]
            encrypted_bytes[i] = data_bytes[i] ^ key_byte
            
        return bytes(encrypted_bytes)

    # --- R-LWE Key Exchange Handler ---

    def _handle_key_exchange(self):
        """Executes the R-LWE key exchange protocol (Client is the initiator of A)."""
        print("[*] Initiating R-LWE Key Exchange...")
        
        try:
            # 1. Receive Server's initial 225 response (Contains A_params)
            response = self._receive_response()
            if not response or not response.startswith('225'): raise Exception("Did not receive A_params hint.")
            
            # 2. Receive A from Server
            server_A_list = self._recv_json()
            if server_A_list is None: raise Exception("Did not receive A.")
            A = np.array(server_A_list, dtype=int)
            
            # 3. Client KeyGen
            s_A, B_A = KeyGen(A)

            # 4. Send Client's Public Key B_A
            self._send_command("B_A SENT") # Placeholder command for logging
            self._send_json(B_A.tolist())
            response = self._receive_response()
            if not response or not response.startswith('226'): raise Exception("Did not receive B_B hint.")
            
            # 5. Receive Server's Public Key B_B
            server_B_B_list = self._recv_json()
            if server_B_B_list is None: raise Exception("Did not receive B_B.")
            B_B = np.array(server_B_B_list, dtype=int)
            
            # 6. Calculate raw secret v_A and generate hint h
            raw_v_A = PO.poly_mul(B_B, s_A)
            h = HelpRec(raw_v_A)

            # 7. Send Client's Hint h
            self._send_command("H SENT") # Placeholder command for logging
            self._send_json(h.tolist())
            response = self._receive_response()
            if not response or not response.startswith('227'): raise Exception("Did not receive final K hint.")
            
            # 8. Client computes Shared Secret K
            K_A = Rec(raw_v_A, h)
            self.shared_key = K_A
            
            # 9. Final 230 response
            response = self._receive_response()
            if not response or not response.startswith('230'): raise Exception("Server key finalization failed.")

            print(f"[KEY SUCCESS] Shared Key K established (First 8 bits: {K_A[:8]})")
            return True
        
        except Exception as e:
            print(f"[KEY EXCHANGE FAILED] {e}")
            self.close()
            return False

    def _parse_pasv_response(self, response):
        """Parses the 227 PASV response to get IP and Port."""
        if not response or not response.startswith('227'):
            return False
            
        match = re.search(r'\((\d+),(\d+),(\d+),(\d+),(\d+),(\d+)\)', response)
        
        if match:
            parts = [int(p) for p in match.groups()]
            ip = f"{parts[0]}.{parts[1]}.{parts[2]}.{parts[3]}"
            port = (parts[4] * 256) + parts[5]
            
            self.data_ip = ip
            self.data_port = port
            return True
        else:
            print("[ERROR] Failed to parse PASV response.")
            return False

    def _handle_list(self, arg):
        """Handles the LIST command (DECRYPTING directory listing)."""
        self._send_command("PASV")
        pasv_response = self._receive_response()
        if not self._parse_pasv_response(pasv_response): return
        
        self._send_command(f"LIST {arg}")
        response = self._receive_response()

        if response and response.startswith('150'):
            data_socket = self._open_data_connection()
            if data_socket:
                try:
                    encrypted_data = b''
                    while True:
                        chunk = data_socket.recv(BUFFER_SIZE)
                        if not chunk: break
                        encrypted_data += chunk
                    
                    # DECRYPT the entire received block
                    decrypted_listing = self.xor_data(encrypted_data).decode('utf-8')
                    
                    print("\n--- Decrypted Directory Listing ---")
                    sys.stdout.write(decrypted_listing)
                    print("-----------------------------------\n")
                except Exception as e:
                    print(f"[TRANSFER ERROR] Failed during LIST data transfer/decryption: {e}")
                finally:
                    data_socket.close()
            
            self._receive_response()


    def _handle_retr(self, remote_filename):
        """Handles the RETR command (DECRYPTING file download)."""
        local_filename = os.path.basename(remote_filename)
        
        self._send_command("PASV")
        pasv_response = self._receive_response()
        if not self._parse_pasv_response(pasv_response): return
        
        self._send_command(f"RETR {remote_filename}")
        response = self._receive_response()

        if response and response.startswith('150'):
            data_socket = self._open_data_connection()
            if data_socket:
                temp_path = local_filename + ".tmp"
                try:
                    print(f"[*] Receiving and decrypting file: {local_filename}...")
                    with open(temp_path, 'wb') as f:
                        while True:
                            encrypted_data = data_socket.recv(BUFFER_SIZE)
                            if not encrypted_data: break
                            
                            # DECRYPT each chunk before writing
                            decrypted_data = self.xor_data(encrypted_data)
                            f.write(decrypted_data)
                            
                    os.rename(temp_path, local_filename)
                    print(f"[*] Successfully downloaded and decrypted file: {local_filename}")
                except Exception as e:
                    print(f"[TRANSFER ERROR] Failed during RETR data transfer/decryption: {e}")
                    if os.path.exists(temp_path): os.remove(temp_path)
                finally:
                    data_socket.close()
            
            self._receive_response()

    def _handle_stor(self, local_filename):
        """Handles the STOR command (ENCRYPTING file upload)."""
        if not os.path.exists(local_filename) or os.path.isdir(local_filename):
            print(f"[CLIENT ERROR] Local file not found: {local_filename}")
            return
            
        remote_filename = os.path.basename(local_filename)

        self._send_command("PASV")
        pasv_response = self._receive_response()
        if not self._parse_pasv_response(pasv_response): return
        
        self._send_command(f"STOR {remote_filename}")
        response = self._receive_response()

        if response and response.startswith('150'):
            data_socket = self._open_data_connection()
            if data_socket:
                try:
                    print(f"[*] Reading and encrypting file: {local_filename}...")
                    with open(local_filename, 'rb') as f:
                        while True:
                            data = f.read(BUFFER_SIZE)
                            if not data: break
                            # ENCRYPT each chunk before sending
                            encrypted_data = self.xor_data(data)
                            data_socket.sendall(encrypted_data)
                    print(f"[*] Successfully encrypted and uploaded file: {remote_filename}")
                except Exception as e:
                    print(f"[TRANSFER ERROR] Failed during STOR data transfer/encryption: {e}")
                finally:
                    data_socket.close()
            
            self._receive_response()


    def run(self):
        """Main client interaction loop."""
        if not self.control_socket:
            print("[FATAL] Cannot run. No control connection.")
            return

        while True:
            try:
                command_line = input("S-FTP> ").strip()
                if not command_line: continue

                parts = command_line.split()
                command = parts[0].upper()
                arg = parts[1] if len(parts) > 1 else ""

                if command == 'EXIT':
                    self._send_command("QUIT")
                    self._receive_response()
                    break
                elif command == 'LIST':
                    self._handle_list(arg)
                elif command == 'RETR':
                    self._handle_retr(arg)
                elif command == 'STOR':
                    self._handle_stor(arg)
                else:
                    self._send_command(command_line)
                    if not self._receive_response(): break

            except EOFError:
                print("\nReceived EOF, closing connection.")
                self._send_command("QUIT")
                self._receive_response()
                break
            except Exception as e:
                print(f"[UNEXPECTED ERROR] {e}")
                break

        self.close()

    def close(self):
        """Closes the control socket."""
        if self.control_socket:
            self.control_socket.close()
            print("[*] Control socket closed.")

if __name__ == '__main__':
    client = FTPClient()
    if client.connect():
        client.run()