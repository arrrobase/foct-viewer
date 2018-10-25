import time
import sys
import numpy as np
import os

from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QPushButton, QTextEdit, QVBoxLayout, QWidget


class Worker(QObject):
    """
    Must derive from QObject in order to emit signals, connect slots to other signals, and operate in a QThread.
    """

    sig_start = pyqtSignal()
    sig_progress = pyqtSignal(int)
    sig_done = pyqtSignal()

    def __init__(self, parent, foct_path=None, ssada_path=None):
        super().__init__()
        self.parent = parent

        self.foct_path = foct_path
        self.ssada_path = ssada_path

        self.foct_data = None
        self.ssada_data = None

        self._abort = False

    @pyqtSlot()
    def load(self):
        """Loads foct data in separate thread to not hang GUI"""

        self.sig_start.emit()

        if self.foct_path is not None:
            # self.foct_data = np.fromfile(str(self.foct_path), dtype='float32')
            self.foct_data = self.chunker(self.foct_path)

        if self.ssada_path is not None:
            # self.ssada_data = np.fromfile(str(self.ssada_path), dtype='float32')
            self.ssada_data = self.chunker(self.ssada_path)

        self.reshape()

        self.sig_done.emit()

    def chunker(self, fp, nchunks=100):
        """Loads file in chunks for progress bar"""

        fsize = os.path.getsize(str(fp))

        # if not a divisor, find nearest
        if not fsize % nchunks == 0:

            def factors(n):
                return set(x for tup in ([i, n//i]
                            for i in range(1, int(n**0.5)+1) if n % i == 0) for x in tup)

            nchunks = min(factors(fsize), key=lambda x:abs(x-nchunks))

        chunks = []

        with open(str(fp), 'rb') as f:
            self.sig_progress.emit(0)
            for i in range(nchunks):
                temp = np.frombuffer(f.read(fsize // nchunks), dtype='float32')
                chunks.append(temp)

                pct = 100 // nchunks * (i + 1)
                self.sig_progress.emit(pct)

        chunked = np.concatenate(chunks)

        # if not easy divisor, pct might not get to 100
        if pct != 100:
            self.sig_progress.emit(100)

        return chunked

    def reshape(self):
        """Reshapes the loaded data"""
        # TODO: reshape based on size rather than reading filename (more robust) but won't know 3vs6vs4.5

        size_map = {
            False: (304, 304, 640),
            True: (400, 400, 640),
        }

        size = self.foct_path.parent.stem.split('_')[-1]
        if size[-2:] == 'HD':
            hd = True
            size = size[-3]
        else:
            hd = False
            size = size[-1]

        foct_r = size_map[hd]
        ssada_r = foct_r[:2] + (int(foct_r[2] / 4),)

        if self.foct_data is not None:
            self.foct_data = self.foct_data.reshape(foct_r)
            self.foct_data = self.foct_data / self.foct_data.max() * 255
            self.foct_data = self.foct_data.astype('uint8')

        if self.ssada_data is not None:
            self.ssada_data = self.ssada_data.reshape(ssada_r)
            self.ssada_data = self.ssada_data / self.ssada_data.max() * 255
            self.ssada_data = self.ssada_data.astype('uint8')
