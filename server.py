import socket
import threading
import os
import random
import json
import sys

import numpy as np

# Import constants and R-LWE functions
from lwe_constants import N, Q, K_ERROR
from ftp_constants import FTP_PORT, SERVER_HOST, BUFFER_SIZE, SERVER_ROOT, USER_CREDENTIALS

from r_lwe import KeyGen, HelpRec, Rec
from poly_ops import PO # Get the pre-instantiated PolyOps object

if not os.path.exists(SERVER_ROOT):
    os.makedirs(SERVER_ROOT)

class FTP_Session:
    """Manages a single client session with R-LWE state."""
    def __init__(self, conn, addr):
        self.conn = conn
        self.addr = addr
        self.authenticated = False
        self.current_user = None
        self.cwd = os.path.abspath(SERVER_ROOT)
        self.data_socket = None
        self.shared_key = None 
        self.key_exchange_complete = False

    def send_response(self, code, message):
        """Sends an FTP response back to the client."""
        response = f"{code} {message}\r\n"
        sys.stdout.write(f"-> {self.addr[0]}:{self.addr[1]} | {response}")
        try:
            self.conn.sendall(response.encode('utf-8'))
        except socket.error:
            self.conn.close()

    # Communication helpers for R-LWE (JSON serialization of polynomials)
    def send_json(self, data):
        """Sends structured data (polys) over the control channel."""
        try:
            json_data = json.dumps(data)
            self.conn.sendall(json_data.encode('utf-8') + b'\r\n')
        except socket.error as e:
            print(f"[ERROR] Failed to send JSON: {e}")
            self.conn.close()

    def recv_json(self):
        """Receives structured data (polys) from the control channel."""
        data = b''
        while True:
            try:
                chunk = self.conn.recv(1)
                if chunk == b'\n': break
                if chunk != b'\r': data += chunk
            except socket.error: return None # Connection closed
        
        try:
            return json.loads(data.decode('utf-8'))
        except json.JSONDecodeError:
            print(f"[ERROR] JSON decoding failed.")
            return None

    def start_data_connection(self):
        """Establishes the data connection socket (passive mode)."""
        if not self.data_socket:
            self.send_response(425, "No data connection established (use PASV).")
            return None
        
        try:
            data_conn, data_addr = self.data_socket.accept()
            print(f"[DATA CONNECTED] Client {data_addr[0]}:{data_addr[1]} for transfer.")
            return data_conn
        except socket.error as e:
            print(f"[ERROR] Failed to accept data connection: {e}")
            return None
        finally:
            if self.data_socket:
                 self.data_socket.close()
                 self.data_socket = None

    def xor_data(self, data_bytes):
        """Performs XOR encryption/decryption using the repeating R-LWE key (K)."""
        if self.shared_key is None: raise ValueError("Shared key not established.")

        key_stream = np.array(self.shared_key, dtype=np.uint8)
        key_len = len(key_stream)
        
        encrypted_bytes = bytearray(len(data_bytes))
        
        for i in range(len(data_bytes)):
            key_byte = key_stream[i % key_len]
            encrypted_bytes[i] = data_bytes[i] ^ key_byte
            
        return bytes(encrypted_bytes)

    # --- R-LWE Key Exchange Handler ---

    def handle_key_exchange(self):
        """Executes the R-LWE key exchange protocol."""
        self.send_response(220, "Initiating R-LWE Key Exchange.")

        try:
            # 1. Server samples m and sends it to server
            m = PO.sample_m()

            # 2. Send m to client
            self.send_response(225, "Sending public m")
            self.send_json(m.tolist())

            # 2. Server generates public key P_A and secret key s_A
            s_A, P_A = KeyGen(m)

            # 2a.Confirm client m same as sent m

            client_m_list = self.recv_json()
            if client_m_list is None: raise Exception("Did not receive confirmation m.")
            
            m_confirm = np.array(client_m_list, dtype=int)
            if m_confirm is None: raise Exception("Did not receive m confirmation.")
            if m_confirm.any() != m.any(): raise Exception("m confirmation different from m.")

            print("" \
            "public m sharing done." \
            "")

            # 3. Send public key P_A to client
            self.send_response(226, "m public confirmed. Sending public key P_A to client.")
            self.send_json(P_A.tolist())
            
            # 4. Receive intermediary public key P_B
            client_P_B_list = self.recv_json()
            if client_P_B_list is None: raise Exception("Did not receive B_A.")
            P_B = np.array(client_P_B_list, dtype=int)

            # 5. Calculate intermediary shared key K_A
            spB = PO.poly_mul(s_A, P_B)
            e   = PO.sample_small(k=K_ERROR)
            spe = spB + 2 * e
            K_A = PO.reduce_mod_q(spe)

            self.send_response(227, "Received P_B. Sending reclamation hint.")
            h = PO.sample_binary()
            self.send_json(h.tolist())
            
            # 6. Calculate shared key SK_A
            SK_A = Rec(K_A, h)
            
            self.shared_key = SK_A
            self.key_exchange_complete = True
            
            self.send_response(230, "Key Exchange successful. Ready for USER/PASS.")
            print(f"[KEY SUCCESS] Shared Key K established.")
            
        except Exception as e:
            print(f"[KEY EXCHANGE FAILED] {e}")
            self.send_response(500, "Key Exchange Failed. Disconnecting.")
            self.conn.close()
            return

    # --- Command Handlers (omitted for brevity, assume they call xor_data correctly) ---
    def check_auth_and_key(self):
        if not self.key_exchange_complete:
            self.send_response(503, "Key Exchange required before authentication.")
            return False
        if not self.authenticated:
            self.send_response(530, "Not logged in.")
            return False
        return True

    def handle_user(self, username):
        if not self.key_exchange_complete: self.send_response(503, "Key Exchange must complete first."); return
        if username in USER_CREDENTIALS: self.current_user = username; self.send_response(331, "Username okay, need password.")
        else: self.send_response(530, "Not logged in. Invalid username.")

    def handle_pass(self, password):
        if self.current_user and USER_CREDENTIALS.get(self.current_user) == password:
            self.authenticated = True; self.send_response(230, "User logged in, proceed."); print(f"[AUTH SUCCESS] User: {self.current_user}")
        else: self.send_response(530, "Not logged in. Incorrect password."); self.current_user = None

    def handle_pasv(self):
        if not self.check_auth_and_key(): return
        if self.data_socket: self.data_socket.close()
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = random.randint(49152, 65535)
        self.data_socket.bind((SERVER_HOST, port)); self.data_socket.listen(1)
        ip_parts = SERVER_HOST.split('.'); p1 = port // 256; p2 = port % 256
        data_info = f"{ip_parts[0]},{ip_parts[1]},{ip_parts[2]},{ip_parts[3]},{p1},{p2}"
        self.send_response(227, f"Entering Passive Mode ({data_info})"); print(f"[PASV] Opened data port {port}")

    def handle_list(self):
        if not self.check_auth_and_key(): return
        full_path = self.cwd
        if not os.path.isdir(full_path): self.send_response(550, "CWD is not a directory."); return
        self.send_response(150, "Opening ENCRYPTED data connection for file list.")
        data_conn = self.start_data_connection()
        if data_conn:
            try:
                files = os.listdir(full_path); listing = ""
                for item in files:
                    item_path = os.path.join(full_path, item); size = os.path.getsize(item_path)
                    type_char = 'd' if os.path.isdir(item_path) else '-'; listing += f"{type_char}rw-rw-r-- 1 ftp ftp {size:>10} Jan 01 00:00 {item}\r\n"
                encrypted_listing = self.xor_data(listing.encode('utf-8')); data_conn.sendall(encrypted_listing)
                self.send_response(226, "Encrypted Directory send successful.")
            except Exception as e: self.send_response(550, f"Failed to list directory: {e}")
            finally: data_conn.close()

    def handle_retr(self, filename):
        if not self.check_auth_and_key(): return
        full_path = os.path.join(self.cwd, filename)
        if not os.path.exists(full_path) or os.path.isdir(full_path): self.send_response(550, "File not found or is a directory."); return
        self.send_response(150, f"Opening ENCRYPTED data connection for {filename}.")
        data_conn = self.start_data_connection()
        if data_conn:
            try:
                with open(full_path, 'rb') as f:
                    while True:
                        data = f.read(BUFFER_SIZE);
                        if not data: break
                        encrypted_data = self.xor_data(data); data_conn.sendall(encrypted_data)
                self.send_response(226, "Encrypted Transfer complete.")
            except Exception as e: self.send_response(550, f"Failed to retrieve file: {e}")
            finally: data_conn.close()

    def handle_stor(self, filename):
        if not self.check_auth_and_key(): return
        full_path = os.path.join(self.cwd, filename)
        self.send_response(150, f"Opening ENCRYPTED data connection for {filename}.")
        data_conn = self.start_data_connection()
        if data_conn:
            temp_path = full_path + ".tmp"
            try:
                with open(temp_path, 'wb') as f:
                    while True:
                        encrypted_data = data_conn.recv(BUFFER_SIZE)
                        if not encrypted_data: break
                        decrypted_data = self.xor_data(encrypted_data)
                        f.write(decrypted_data)
                os.rename(temp_path, full_path); self.send_response(226, "Encrypted Transfer complete.")
            except Exception as e: self.send_response(550, f"Failed to store file: {e}"); 
            finally: data_conn.close()

    def handle_quit(self):
        self.send_response(221, "Goodbye."); self.conn.close(); return True

    def handle_command(self, command_line):
        parts = command_line.strip().split(); command = parts[0].upper(); arg = parts[1] if len(parts) > 1 else ""
        if command == 'USER': self.handle_user(arg)
        elif command == 'PASS': self.handle_pass(arg)
        elif command == 'PASV': self.handle_pasv()
        elif command == 'LIST': self.handle_list()
        elif command == 'RETR': self.handle_retr(arg)
        elif command == 'STOR': self.handle_stor(arg)
        elif command == 'QUIT': return self.handle_quit()
        else: self.send_response(502, f"Command {command} not implemented.")
        return False

    def run(self):
        self.handle_key_exchange()
        if not self.key_exchange_complete: return

        try:
            while True:
                data = self.conn.recv(BUFFER_SIZE).decode('utf-8')
                if not data: break
                commands = data.split('\r\n')
                for command_line in commands:
                    if command_line and self.handle_command(command_line): return
        except ConnectionResetError:
            print(f"[ERROR] Client {self.addr} forcefully disconnected.")
        finally:
            self.conn.close()

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Ensure server root exists
    if not os.path.exists(SERVER_ROOT):
        os.makedirs(SERVER_ROOT)

    try:
        server_socket.bind((SERVER_HOST, FTP_PORT))
        server_socket.listen(5)
        print(f"[*] Secure FTP Server listening on {SERVER_HOST}:{FTP_PORT}")

        while True:
            conn, addr = server_socket.accept()
            print(f"[NEW CONNECTION] Connected to {addr[0]}:{addr[1]}")
            client_thread = threading.Thread(target=FTP_Session(conn, addr).run)
            client_thread.start()

    except Exception as e:
        print(f"[FATAL ERROR] Server crashed: {e}")
    finally:
        server_socket.close()

if __name__ == '__main__':
    start_server()