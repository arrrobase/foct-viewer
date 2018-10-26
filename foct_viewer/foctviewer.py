import sys
import traceback as tb
from pathlib import Path

import cv2
import numpy as np
import pkg_resources
import pyqtgraph as pg

from PyQt5.QtCore import (Qt, pyqtSignal, QThread)
from PyQt5.QtGui import QIcon, QCursor
from PyQt5.QtWidgets import (QAction, QApplication, QDesktopWidget, QDialog, QFileDialog, QProgressBar, QLabel,
                             QMainWindow, QToolBar, QVBoxLayout, QHBoxLayout, QErrorMessage, QStatusBar,
                             QWidget, QDockWidget, QCheckBox, QStyle, QStyleOptionButton, QDoubleSpinBox)

from numpy.polynomial import Chebyshev as T
from pyqtgraph import ptime as ptime
from scipy.io import loadmat

from foct_viewer import LoaderWorker


# setup exception handler
# PyQt4 would print errors to IDE console and suppress them, but
# PyQt5 when run in an IDE just terminates without printing a traceback
def except_hook(type, value, traceback, frame):
    """
    Catches and logs exceptions without crashing pyqt

    :param type:
    :param value:
    :param traceback:
    :param error_dialog:
    :return:
    """
    # logger.error(''.join(tb.format_exception(*(type, value, traceback))))
    frame.show_traceback(type, value, traceback)

    sys.__excepthook__(type, value, traceback)


class FoctViewer(QMainWindow):
    """Create the main window that stores all of the widgets necessary for the application."""

    def __init__(self, parent=None):
        """Initialize the components of the main window."""
        super(FoctViewer, self).__init__(parent)

        self.resize(1024, 768)
        self.setWindowTitle('FoctViewer')
        window_icon = pkg_resources.resource_filename('foct_viewer.images',
                                                      'ic_insert_drive_file_black_48dp_1x.png')
        self.setWindowIcon(QIcon(window_icon))

        self.error_dialog = QErrorMessage()
        self.error_dialog.setFixedWidth(600)
        self.error_dialog.setMinimumHeight(300)

        self.menu_bar = self.menuBar()
        self.about_dialog = AboutDialog()

        self.status_bar = self.create_status_bar()
        self.status_bar.showMessage('Ready', 5000)

        self.central_widget = MainWidget(self)
        self.setCentralWidget(self.central_widget)

        self.file_menu()
        self.help_menu()

        # left line dock widget
        self.ui_dock = LinesDock('Layers', self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.ui_dock)

        # self.tool_bar_items()
        self.show()

    def create_status_bar(self):
        """
        Creates the status bar
        """
        statusbar = QStatusBar()
        statusbar.setSizeGripEnabled(False)

        statusbar.ind_label = QLabel('0 ')
        statusbar.addPermanentWidget(statusbar.ind_label)
        statusbar.ind_label.hide()

        self.create_loading_bar(statusbar)

        self.setStatusBar(statusbar)
        statusbar.showMessage('Welcome', 2000)

        return statusbar

    def create_loading_bar(self, statusbar):
        """Creates loading bar
        """
        STYLE = """
        QProgressBar{
            height: 10px;
            border: 1px solid grey;
        }

        QProgressBar::chunk {
            background-color: darkblue;
        }
        """

        self.load_bar = QProgressBar(self)
        statusbar.addPermanentWidget(self.load_bar)
        self.load_bar.setMaximum(100)
        self.load_bar.setStyleSheet(STYLE)
        self.load_bar.setTextVisible(False)

        self.load_bar.hide()

    def file_menu(self):
        """Create a file submenu with an Open File item that opens a file dialog."""

        ### FILE ###

        self.menu_file = self.menu_bar.addMenu('File')

        menu_open = QAction('Open OCTA', self)
        menu_open.setStatusTip('Open a file into FoctViewer.')
        menu_open.setShortcut('CTRL+O')
        menu_open.triggered.connect(self.on_file_open)

        menu_open_seg = QAction('Open seg', self)
        menu_open_seg.setStatusTip('Open a segment file.')
        menu_open_seg.triggered.connect(self.on_file_open_seg)

        menu_exit = QAction('Exit Application', self)
        menu_exit.setStatusTip('Exit the application.')
        menu_exit.setShortcut('CTRL+Q')
        menu_exit.triggered.connect(lambda: QApplication.quit())

        self.menu_file.addAction(menu_open)
        self.menu_file.addAction(menu_open_seg)
        self.menu_file.addAction(menu_exit)

        ### VIEW ###

        self.menu_view = self.menu_bar.addMenu('View')

        view_histogram = QAction('Toggle hist', self)
        view_histogram.setStatusTip('Hide or show the LUT levels histogram.')
        view_histogram.setShortcut('CTRL+H')
        view_histogram.triggered.connect(self.central_widget.toggle_histogram)

        self.menu_view.addAction(view_histogram)

        ### OPTIONS ###

        self.menu_options = self.menu_bar.addMenu('Options')

        self.options_ssada = QAction('Autoload ssada', self)
        self.options_ssada.setStatusTip('Whether or not to load SSADA when loading FOCT.')
        self.options_ssada.setCheckable(True)
        self.options_ssada.setChecked(False)

        self.options_seg = QAction('Autoload seg', self)
        self.options_seg.setStatusTip('Whether or not to load segment when loading FOCT.')
        self.options_seg.setCheckable(True)
        self.options_seg.setChecked(True)

        self.menu_options.addAction(self.options_ssada)
        self.menu_options.addAction(self.options_seg)

    def help_menu(self):
        """Create a help submenu with an About item that opens an about dialog."""
        self.help_sub_menu = self.menu_bar.addMenu('Help')

        self.about_action = QAction('About', self)
        self.about_action.setStatusTip('About the application.')
        # self.about_action.setShortcut('CTRL+H')
        self.about_action.triggered.connect(lambda: self.about_dialog.exec_())

        self.help_sub_menu.addAction(self.about_action)

    def tool_bar_items(self):
        """Create a tool bar for the main window."""
        self.tool_bar = QToolBar()
        self.addToolBar(Qt.TopToolBarArea, self.tool_bar)
        self.tool_bar.setMovable(False)

        open_icon = pkg_resources.resource_filename('foct_viewer.images',
                                                    'ic_open_in_new_black_48dp_1x.png')
        tool_bar_open_action = QAction(QIcon(open_icon), 'Open File', self)
        tool_bar_open_action.triggered.connect(self.on_file_open)

        self.tool_bar.addAction(tool_bar_open_action)

    def on_file_open(self):
        """Open a QFileDialog to allow the user to open a file into the application."""
        filename, accepted = QFileDialog.getOpenFileName(self, 'Open File',
                                                         filter='Optovue (*.foct);;'\
                                                                'All files (*)')

        if accepted:
            filename = Path(filename)
            self.central_widget.open_oct(filename)

    def on_file_open_seg(self):
        """Open a QFileDialog to allow the user to open a file into the application."""
        filename, accepted = QFileDialog.getOpenFileName(self, 'Open File',
                                                         filter='Segment (*.mat);;'\
                                                                'All files (*)')

        if accepted:
            filename = Path(filename)
            self.central_widget.open_seg(filename)

    def show_traceback(self, type, value, traceback):
        """
        Formatting error dialog
        """
        self.error_dialog.showMessage('<br><br>'.join(tb.format_exception(*(type, value, traceback))))


class MainWidget(pg.ImageView):
    """Main foct viewer widget"""

    sig_data_changed = pyqtSignal()
    sig_lines_changed = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize the components of the viewer widget."""
        super(MainWidget, self).__init__(parent)
        self.frame = parent

        self.status_bar = self.frame.status_bar

        # set cursor
        # cursor_img = pkg_resources.resource_filename('foct_viewer.images',
        #                                               'cursor-cross.png')
        # pixmap = QPixmap(cursor_img)
        # crosshair = QCursor(pixmap)
        crosshair = QCursor()
        crosshair.setShape(4)
        self.view.setCursor(Qt.CrossCursor)

        self.toggle_histogram(ev=None, hide=True)
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()

        self.view.invertY(False)

        # plot items for seg data
        num_lines = 8
        self.line_plots = [pg.PlotCurveItem() for i in range(num_lines)]

        for line in self.line_plots:
            self.view.addItem(line)

        # attributes
        self.foct_path = None
        self.foct_data = None
        self.ssada_data = None
        self.seg_data = None
        self.play_stop = None
        self.lims = None
        self.path_label = None
        self.step = 24
        self.line_drawing = 0

        self.sigTimeChanged.connect(self.draw_lines)
        self.sig_data_changed.connect(self.on_data_changed)
        self.sig_lines_changed.connect(self.on_lines_changed)

        # monkey patch wheelevent from viewbox
        self.view.wheelEvent = self.wheelEvent

        # moneky patch mouseclick event while keeping old
        self.view_old_mouseClickEvent = self.view.mouseClickEvent
        self.view.mouseClickEvent = self.mouseClickEvent

        # moneky patch mouseclick event while keeping old
        self.view_old_mouseDragEvent = self.view.mouseDragEvent
        self.view.mouseDragEvent = self.mouseDragEvent

    def setCurrentIndex(self, ind):
        """add time change sig to super"""
        super().setCurrentIndex(ind)
        (ind, time) = self.timeIndex(self.timeLine)
        self.sigTimeChanged.emit(ind, time)

    def toggle_histogram(self, ev, hide=None):
        """Toggles the viz of the histogram"""
        if hide is not None:
            if hide:
                self.ui.histogram.hide()
            else:
                self.ui.histogram.show()

        else:
            if self.ui.histogram.isHidden():
                self.ui.histogram.show()
            else:
                self.ui.histogram.hide()

    def open_oct(self, filename=None):
        """Opens an foct file and associated files"""
        foct_path = filename
        try:
            ssada_path = list(foct_path.parent.glob('*.ssada'))[0]
        except IndexError:
            ssada_path = None
        try:
            seg_path = list(foct_path.parent.glob('*seg*.mat'))[0]
        except IndexError:
            seg_path = None

        if not self.frame.options_ssada.isChecked():
            ssada_path = None

        self.load_foct(foct_path, ssada_path)

        if self.frame.options_seg.isChecked():
            self.open_seg(seg_path, force=True)

    def open_seg(self, filename=None, force=False):
        """
        Opens a segment file

        :param filename:
        :param force: if True, ignore the check for FOCT data
        :return:
        """
        seg_path = filename
        if seg_path is None:
            return

        seg_data = loadmat(str(seg_path), appendmat=False)

        try:
            seg_lines = seg_data['ManualCurveData']
            seg_lines = 640 - seg_lines

            if not force:
                if self.foct_data is not None:
                    assert seg_lines.shape[::2] == self.foct_data.shape[:2]
                else:
                    raise AttributeError('Cannot load segment file without FOCT data loaded.')

        except (KeyError, AssertionError):
            raise FileNotFoundError('Cannot load segment file; not expected format.')

        self.load_seg(seg_lines)

    def load_foct(self, foct_path, ssada_path=None):
        """Loads foct and ssada and reshapes, in seperate thread"""

        # TODO: if going from segged HD to segged SD, out of bounds error from trying to draw lines past 304

        # first remove segs and curent
        self.clear()
        self.lims = None
        self.foct_path = foct_path
        # TODO: if loading fails, we'd like the segs to still be there
        self.seg_data = None
        self.draw_lines()

        self.worker = LoaderWorker.Worker(self, foct_path)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        # get progress messages from worker:
        self.worker.sig_start.connect(self.on_worker_start)
        self.worker.sig_done.connect(self.on_worker_done)
        self.worker.sig_progress.connect(self.update_bar)

        self.thread.started.connect(self.worker.load)
        self.thread.start()

    def on_worker_start(self):
        self.frame.status_bar.showMessage('loading oct data...')

    def on_worker_done(self):
        self.frame.status_bar.showMessage('done!')
        self.sig_data_changed.emit()

    def load_seg(self, seg_lines):
        """Loads seg lines"""
        self.seg_data = seg_lines
        self.sig_lines_changed.emit()

    def update_bar(self, pct):
        """Updates progress bar"""
        if pct == 0:
            self.status_bar.ind_label.hide()
            self.frame.load_bar.show()

        self.frame.load_bar.setValue(pct)

        if pct == 100:
            self.frame.load_bar.hide()
            self.status_bar.ind_label.show()

    def on_data_changed(self):
        self.foct_data = self.worker.foct_data
        self.ssada_data = self.worker.ssada_data

        # remove worker thread
        self.thread.quit()
        del(self.worker)

        self.set_level()
        self.setImage(self.foct_data, autoLevels=False)
        self.set_lims()
        self.frame.ui_dock.spinbox_step.setRange(0, self.foct_data.shape[-1])

        self.status_bar.clearMessage()
        if self.path_label is None:
            self.path_label = QLabel(str(self.foct_path))
            self.status_bar.addWidget(self.path_label)
        else:
            self.path_label.setText(str(self.foct_path))

    def on_lines_changed(self):
        if self.seg_data is not None:
            self.draw_lines()

    def set_lims(self):
        """Sets the zoom
        """
        max_loc = np.argmax(self.foct_data[150].mean(axis=0))
        upper_thresh = max_loc + 200
        lower_thresh = max_loc - 150

        self.lims = (lower_thresh, upper_thresh)

        self.view.setYRange(self.lims[0], self.lims[1])

    def set_level(self):
        self.status_bar.showMessage('calculating levels...')
        f = cv2.calcHist([self.foct_data.flatten()], [0], None, [256], [0, 256])
        f = f[:, 0].astype(int)

        # fit vars
        # only want to fit right of max peak
        max_loc = f.argmax()
        f_fmax = f[max_loc:]
        x_max = np.linspace(max_loc, 255, num=len(f_fmax))

        # fit a polynomial of order
        order = 50
        poly_fit = T.fit(x_max, f_fmax, order)

        # get real roots
        roots = np.real_if_close(poly_fit.deriv(2).roots())
        roots = roots[np.isreal(roots)].real
        # get roots to right of max peak
        roots = roots[((roots >= max_loc) & (roots < 256))]

        up_thresh = int(roots[((poly_fit(roots) / f.max()) < 0.15).argmax()])

        # get lower thresh
        r_f = np.arange(len(f)) > f.argmax()
        up_f = (f / f.max() < 0.002)

        low_thresh = (r_f & up_f).argmax()

        self.setLevels(up_thresh, low_thresh)

        self.status_bar.clearMessage()
        self.status_bar.showMessage('done!', 3000)

    def draw_lines(self, ind=None):

        ind = self.currentIndex

        self.status_bar.ind_label.setText('{} '.format(int(ind) + 1))

        if self.seg_data is None:
            for idx, line in enumerate(self.line_plots):
                line.setData(None)
            return

        for to_draw, (idx, line) in zip(self.show_lines, enumerate(self.line_plots)):
            if to_draw:
                line.setData(self.seg_data[:, idx, ind])
            else:
                line.setData(None)

    def wheelEvent(self, ev, axis=None):
        """Overrides wheel event"""
        try:
            # a single mouse tick is 15 degrees
            # delta is reported in 1/8ths of degree
            # so a single mouse ticks = 120 deltas
            # then flip so scroll down increases index
            ticks = int(ev.delta() / -120)
        except AttributeError:
            return

        if self.foct_data is not None:
            self.jumpFrames(self.step * ticks)

    def mouseClickEvent(self, ev):
        """Monkey patch viewbox mouse click event to steal midbutton"""
        if ev.button() & Qt.MidButton:
            if self.lims is not None:
                # set xrange first to recenter
                self.view.setXRange(0, self.foct_data.shape[0])
                self.view.setYRange(self.lims[0], self.lims[1])

            ev.accept()

        else:
            self.view_old_mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        """Monkey patch viewbox mouse drag event to steal left button drag"""
        if ev.button() & Qt.LeftButton:
            if ev.isFinish():
                pass

            else:
                # TODO: allow drawing when no segdata
                if self.foct_data is not None and self.seg_data is not None:

                    last = self.view.mapToView(ev.lastPos())
                    last_x = int(last.x())

                    p = self.view.mapToView(ev.pos())

                    x = int(p.x())
                    y = int(p.y())

                    if 0 <= x < self.foct_data.shape[0]:

                        dif = last_x - x
                        if abs(dif) > 0:
                            for i in range(0, dif, int(dif / abs(dif))):
                                self.seg_data[x + i, self.line_drawing, self.currentIndex] = y

                        self.sig_lines_changed.emit()

            ev.accept()

        else:
            self.view_old_mouseDragEvent(ev, axis=axis)

    def evalKeyState(self):
        # TODO: while loading, arrow keys cause error
        if len(self.keysPressed) == 1:
            key = list(self.keysPressed.keys())[0]
            if key == Qt.Key_Right:
                self.play(20)
                self.jumpFrames(1)
                self.lastPlayTime = ptime.time() + 0.2  ## 2ms wait before start
                                                        ## This happens *after* jumpFrames, since it might take longer than 2ms
            elif key == Qt.Key_Left:
                self.play(-20)
                self.jumpFrames(-1)
                self.lastPlayTime = ptime.time() + 0.2
            elif key == Qt.Key_Up:
                self.jumpFrames(-self.step)
            elif key == Qt.Key_Down:
                self.jumpFrames(self.step)
            elif key == Qt.Key_PageUp:
                self.play_stop = self.currentIndex - self.step
                self.play(-59)
            elif key == Qt.Key_PageDown:
                self.play_stop = self.currentIndex + self.step
                self.play(59)
        else:
            self.play(0)

    def timeout(self):
        now = ptime.time()
        dt = now - self.lastPlayTime
        if dt < 0:
            return
        n = int(self.playRate * dt)
        if n != 0:
            self.lastPlayTime += (float(n)/self.playRate)
            if self.currentIndex+n > self.image.shape[0]:
                self.play(0)

            if self.play_stop is not None:
                if n > 0 and self.currentIndex+n >= self.play_stop:
                    self.play(0)
                    self.play_stop = None

                if n < 0 and self.currentIndex+n <= self.play_stop:
                    self.play(0)
                    self.play_stop = None

            self.jumpFrames(n)

    def on_line_checkbox(self, ev):
        """update lines"""
        # TODO: make cleaner
        self.show_lines = [self.frame.ui_dock.line_checkboxes[line].isChecked() for line in self.frame.ui_dock.lines]

        self.draw_lines()

    def on_line_checkbox_text(self, object_name):
        """update lines"""
        for linename, line in self.frame.ui_dock.line_checkboxes.items():
            if linename == object_name:
                line.highlight(True)
                idx = self.frame.ui_dock.lines.index(object_name)
                self.line_drawing = idx
            else:
                line.highlight(False)

    def on_spinbox_step(self, val=24):
        """Update the step"""
        self.step = int(val)


class LinesDock(QDockWidget):
    """test of dock"""

    def __init__(self, name, parent=None):
        super(LinesDock, self).__init__(name, parent)

        self.parent = parent
        self.central_widget = self.parent.central_widget

        # don't allow closing, and only allow left and right docks
        self.setFeatures(QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.main_widget = QWidget(self)
        self.setWidget(self.main_widget)

        self.setup_ui()

    def setup_ui(self):
        """Lays out ui"""
        layout_control_splitter = QVBoxLayout()

        # step size selector
        layout_step_splitter = QHBoxLayout()
        label_step = QLabel('Step: ')

        self.spinbox_step = QDoubleSpinBox()
        self.spinbox_step.setRange(0, 24)
        self.spinbox_step.setSingleStep(1)
        self.spinbox_step.setValue(self.central_widget.step)
        self.spinbox_step.setDecimals(0)
        self.spinbox_step.valueChanged.connect(self.central_widget.on_spinbox_step)

        layout_step_splitter.addWidget(label_step)
        layout_step_splitter.addWidget(self.spinbox_step)

        layout_control_splitter.addLayout(layout_step_splitter)

        self.lines = [
            'Vitreous/ILM',
            'NFL/GCL',
            'IPL/INL',
            'INL/OPL',
            'OPL/ONL',
            'Inner ISOS',
            'Inner RPE',
            'RPE/BM'
        ]

        self.line_checkboxes = {line:ClickCheckBox(line) for line in self.lines}

        for linename in self.lines:
            linebox = self.line_checkboxes[linename]

            linebox.setChecked(True)
            linebox.setObjectName(linename)
            linebox.stateChanged.connect(self.central_widget.on_line_checkbox)
            linebox.sig_text_clicked.connect(self.central_widget.on_line_checkbox_text)
            layout_control_splitter.addWidget(linebox)

        self.line_checkboxes[self.lines[0]].highlight(True)
        self.parent.central_widget.show_lines = [self.line_checkboxes[line].isChecked() for line in self.lines]

        layout_control_splitter.setAlignment(Qt.AlignTop)

        self.main_widget.setLayout(layout_control_splitter)


class ClickCheckBox(QCheckBox):
    """
    subclass of qcheckbox to overwrite hitbutton
    """
    sig_text_clicked = pyqtSignal(str)

    def __init__(self, *args):
        super(ClickCheckBox, self).__init__(*args)

    def hitButton(self, point):
        """only change if checkbox is clicked"""
        style = QStyle.SE_CheckBoxIndicator
        option = QStyleOptionButton()
        self.initStyleOption(option)

        ret = QApplication.style().subElementRect(style, option, self).contains(point)
        # print(ret)
        return ret

    def hitText(self, point):
        """only change if text is clicked"""
        style = QStyle.SE_CheckBoxContents
        option = QStyleOptionButton()
        self.initStyleOption(option)

        ret = QApplication.style().subElementRect(style, option, self).contains(point)
        return ret

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            ev.ignore()
            return

        if self.hitButton(ev.pos()):
            self.setDown(True)
            self.pressed = True
            self.repaint()
            QApplication.flush()
            # self.emitPressed()
            ev.accept()
        elif self.hitText(ev.pos()):
            self.sig_text_clicked.emit(self.objectName())
        else:
            ev.ignore()

    def highlight(self, state):
        """whether or not to highlight text"""
        # STYLE = """
        # QCheckBox::
        # """
        if state:
            self.setStyleSheet('color: red')
        else:
            self.setStyleSheet('')


class AboutDialog(QDialog):
    """Create the necessary elements to show helpful text in a dialog."""

    def __init__(self, parent=None):
        """Display a dialog that shows application information."""
        super(AboutDialog, self).__init__(parent)

        self.setWindowTitle('About')
        help_icon = pkg_resources.resource_filename('foct_viewer.images',
                                                    'ic_help_black_48dp_1x.png')
        self.setWindowIcon(QIcon(help_icon))
        self.resize(300, 150)

        name = QLabel('Foct Viewer, version 0.1\n')
        name.setAlignment(Qt.AlignCenter)

        author = QLabel('Author: Alexander Tomlinson')
        author.setAlignment(Qt.AlignCenter)

        icons = QLabel('Material design icons created by Google')
        icons.setAlignment(Qt.AlignCenter)

        github = QLabel('GitHub: awctomlinson')
        github.setAlignment(Qt.AlignCenter)

        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignVCenter)

        self.layout.addWidget(name)
        self.layout.addWidget(author)
        self.layout.addWidget(icons)
        self.layout.addWidget(github)

        self.setLayout(self.layout)


def main():
    """main function"""
    application = QApplication(sys.argv)
    window = FoctViewer()

    # catch errors into error dialog
    sys.excepthook = lambda x, y, z: except_hook(x, y, z, window)

    desktop = QDesktopWidget().availableGeometry()
    width = (desktop.width() - window.width()) / 2
    height = (desktop.height() - window.height()) / 2
    window.show()
    window.move(width//2, height//2)

    sys.exit(application.exec_())
