import json
from multiprocessing.shared_memory import SharedMemory
import pickle


def decode_index(index_buffer):
    buf = index_buffer.buf
    index_list = buf.tobytes().decode().strip("\x00").split(",")
    if index_list == ['']:
        index_list = []
    return index_list


def write_index(index_buffer, new_index):
    encoded = ','.join(new_index).encode()
    buf = index_buffer.buf
    buf[:len(encoded)] = encoded


def json_pickle_encoder(value):
    try:
        return json.dumps(value).encode()
    except TypeError:
        return pickle.dumps(value, protocol=5)


def json_pickle_decoder(value):
    try:
        decoded = value.decode()
        return json.loads(decoded)
    except UnicodeDecodeError:
        return pickle.loads(value)


class Paper:
    def __init__(self, prefix, index_size=512):
        self.prefix = prefix
        self.index = SharedMemory(
            f"{prefix}index", create=True, size=index_size
        )


class Notepad:
    def __init__(self, prefix):
        self.prefix = prefix
        self._index_buffer = SharedMemory(f"{prefix}index")
        self.index = decode_index(self._index_buffer)

    def _address(self, key):
        return f"{self.prefix}{key}"

    def _update_index(self):
        self._index_buffer.close()
        self._index_buffer = SharedMemory(f"{self.prefix}index")
        self.index = decode_index(self._index_buffer)

    def __setitem__(self, key, value, encoder=json_pickle_encoder):
        encoded = json_pickle_encoder(value)
        size = len(encoded)
        try:
            block = SharedMemory(self._address(key), create=True, size=size)
        except FileExistsError:
            old_block = SharedMemory(self._address(key))
            old_block.unlink()
            old_block.close()
            block = SharedMemory(self._address(key), create=True, size=size)
        block.buf[:] = encoded
        block.close()
        self.index.append(key)
        self.index = list(set(self.index))
        write_index(self._index_buffer, self.index)

    def __getitem__(self, key, decoder=json_pickle_decoder):
        try:
            block = SharedMemory(self._address(key))
        except FileNotFoundError:
            return None
        stream = block.buf.tobytes()
        block.close()
        return json_pickle_decoder(stream)

    def __delitem__(self, key):
        block = SharedMemory(self._address(key))
        block.unlink()
        block.close()

    def set(self, key, value):
        return self.__setitem__(key, value)

    def get(self, key):
        return self.__getitem__(key)