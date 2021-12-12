import socket
import sys
import os
import time
import numpy as np
import cv2
import threading

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("", 18452))
s.listen(1)

s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s2.bind(("", 18453))
s2.listen(1)

host = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host.bind(("", 18454))
host.listen(1)

lt = time.time()

backbuffer = np.zeros((200, 200, 3), dtype=np.uint8)
buffer = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.imshow("test", cv2.resize(buffer, (800, 800)))
cv2.waitKey(1)

highest_tid = 0

highest_hwid = 0
connection_count = 0

enable_image_display = True

neural_queue = []

last_eye = None
swapped = False

target_stream = None # stream to write parameters to
target_stream_is_valid = False # if the stream is currently valid

def decode_image(data, flipped=False):
    data = np.frombuffer(data, dtype=np.float32)
    data = np.clip(np.reshape(data, (1, 2, 100, 100)) * 255, 0, 255)[0]
    
    new_image = np.zeros((100, 200), dtype=np.float32)
    
    if flipped:
        new_image[:, :100] = np.flip(data[1], 1)
        new_image[:, 100:] = np.flip(data[0], 1)
    else:
        new_image[:, :100] = data[0]
        new_image[:, 100:] = data[1]
    
    new_image = np.stack((new_image, new_image, new_image), axis=-1).astype(np.uint8)
    return new_image

def decode_neural(data):
    data = np.frombuffer(data, dtype=np.float32)
    data = np.reshape(data, (1, 64, 20, 20))[0]
    return data

def reader_thread(s, id):
    global highest_tid
    global highest_hwid
    global connection_count
    global lt
    global backbuffer
    global last_eye
    while True:
        try:
            c, addr = s.accept()
            print("Received connection")
            connection_count = connection_count + 1
            try:

                rid = 0
                while True:
                    def read(length):
                        v = c.recv(length)
                        while len(v) < length:
                            v = v + c.recv(length - len(v))
                        return v
                    
                    def read_int():
                        v = read(4)
                        return int.from_bytes(v, "little")

                    device = read_int() # broken
                    length = read_int()
                    thread_id = read_int()
                    
                    if id == 1 and length == 80000:
                        highest_hwid = (highest_hwid * 0.99) + (device * 0.01) # broken as fuck
                    
                    if thread_id > highest_tid:
                        highest_tid = thread_id
                    
                    if not (102400 == length or 80000 == length):
                        print("Invalid packet!")
                        try:
                            c.close()
                        except:
                            traceback.print_exc()
                        break

                    data = read(length)
                    
                    if length == 80000: # camera frame
                        if enable_image_display:
                            if (id == 0 and (not swapped)) or (id == 1 and swapped):
                                buffer[100:200, :, :] = decode_image(data, flipped=True)
                                backbuffer = buffer.copy()
                            else:
                                buffer[:100, :, :] = decode_image(data)
                    else:
                        if (id == 0 and (not swapped)) or (id == 1 and swapped):
                            if last_eye is not None:
                                neural_queue.append((last_eye, data))
                                if len(neural_queue) > 60 * 4:
                                    neural_queue.pop(0)#del neural_queue[:len(neural_queue)-(60 * 4)]
                        else:
                            last_eye = data

                    rid = rid + 1
            except:
                import traceback
                traceback.print_exc()
                try:
                    c.close()
                except:
                    traceback.print_exc()
                time.sleep(0.1)
        except:
            import traceback
            traceback.print_exc()
        connection_count = connection_count - 1

def server_thread():
    while True:
        try:
            c, addr = host.accept()
            print("Received client connection")
            while True:
                while len(neural_queue) > 0:
                    a, b = neural_queue.pop(0)
                    c.sendall(a)
                    c.sendall(b)
                    
                while len(neural_queue) == 0:
                    time.sleep(0.001)
        except:
            import traceback
            traceback.print_exc()
            time.sleep(0.01)
            
def input_thread():
    global swapped
    while True:
        print("Please enter command (only commands are swap, image and exit)")
        cmd = input()
        if cmd == "swap":
            print("Swapped buffers!")
            swapped = not swapped
            neural_queue.clear()
        elif cmd == "exit":
            quit()
        elif cmd == "image":
            enable_image_display = not enable_image_display

threading.Thread(target=reader_thread, args=(s, 0)).start()
threading.Thread(target=reader_thread, args=(s2, 1)).start()
threading.Thread(target=server_thread, args=()).start()
threading.Thread(target=input_thread, args=()).start()

while True:
    lt = time.time()
    cv2.imshow("test", cv2.resize(backbuffer, (800, 800)))
    cv2.waitKey(16)
    while not enable_image_display:
        time.sleep(0.1)
