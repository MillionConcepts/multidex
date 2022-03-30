import os
import pickle
import stat
import tempfile

from flask_caching import logger
from flask_caching.backends import FileSystemCache


class WindowsFileSystemCache(FileSystemCache):
    def set(self, key, value, timeout=None, mgmt_element=False):
        result = False

        # Management elements have no timeout
        if mgmt_element:
            timeout = 0

        # Don't prune on management element update, to avoid loop
        else:
            self._prune()

        timeout = self._normalize_timeout(timeout)
        filename = self._get_filename(key)
        try:
            fd, tmp = tempfile.mkstemp(
                suffix=self._fs_transaction_suffix, dir=self._path
            )
            with os.fdopen(fd, "wb") as f:
                pickle.dump(timeout, f, 1)
                pickle.dump(value, f, pickle.HIGHEST_PROTOCOL)
            is_new_file = not os.path.exists(filename)
            os.replace(tmp, filename)
            os.chmod(filename, stat.S_IWRITE)
        except (IOError, OSError) as exc:
            logger.error("set key %r -> %s", key, exc)
        else:
            result = True
            logger.debug("set key %r", key)
            # Management elements should not count towards threshold
            if not mgmt_element and is_new_file:
                self._update_count(delta=1)
        return result