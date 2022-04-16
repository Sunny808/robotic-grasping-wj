# version:1.0.1905.9051
import gxipy as gx
import cv2 as cv


# create a device manager
device_manager = gx.DeviceManager()
dev_num, dev_info_list = device_manager.update_device_list()


class VideoCapture:
    def __init__(self,
                 index,
                 width = 640,
                 height = 480
                 ):
        self.index = index
        self.width = width
        self.height = height
        self.cam = device_manager.open_device_by_index(self.index + 1)# open the first device

    def read(self):
        if dev_num == 0:
            print("Number of enumerated devices is 0")
            return

        # exit when the camera is a mono camera
        if self.cam.PixelColorFilter.is_implemented() is False:
            print("This sample does not support mono camera.")
            self.cam.close_device()
            return

        # set continuous acquisition
        self.cam.TriggerMode.set(gx.GxSwitchEntry.OFF)

        # set exposure
        self.cam.ExposureAuto.set(gx.GxAutoEntry.CONTINUOUS)

        # set gain
        self.cam.GainAuto.set(gx.GxAutoEntry.ONCE)

        # set balanceWhite
        self.cam.BalanceWhiteAuto.set(gx.GxAutoEntry.ONCE)

        # set image width height
        self.cam.Width.set(self.width)
        self.cam.Height.set(self.height)

        # get param of improving image quality
        if self.cam.GammaParam.is_readable():
            gamma_value = self.cam.GammaParam.get()
            gamma_lut = gx.Utility.get_gamma_lut(gamma_value)
        else:
            gamma_lut = None
        if self.cam.ContrastParam.is_readable():
            contrast_value = self.cam.ContrastParam.get()
            contrast_lut = gx.Utility.get_contrast_lut(contrast_value)
        else:
            contrast_lut = None
        if self.cam.ColorCorrectionParam.is_readable():
            color_correction_param = self.cam.ColorCorrectionParam.get()
        else:
            color_correction_param = 0

        # start data acquisition
        self.cam.stream_on()

        # get raw image
        raw_image = self.cam.data_stream[0].get_image()
        if raw_image is None:
            print("Getting image failed.")
            return

        # get RGB image from raw image
        rgb_image = raw_image.convert("RGB")
        if rgb_image is None:
            return

        # improve image quality
        rgb_image.image_improvement(color_correction_param, contrast_lut, gamma_lut)

        # create numpy array with data from raw image
        numpy_image = rgb_image.get_numpy_array()
        if numpy_image is None:
            return

        # show acquired image
        img = cv.cvtColor(numpy_image, cv.COLOR_RGB2BGR)
        # cv.imshow('image', img)

        # print height, width, and frame ID of the acquisition image
        '''print("Frame ID: %d   Height: %d   Width: %d"
              % (raw_image.get_frame_id(), raw_image.get_height(), raw_image.get_width()))'''

        # cv.waitKey(0)
        # stop data acquisition
        self.cam.stream_off()
        return img

    # close device
    def release(self):
        self.cam.close_device()


if __name__ == "__main__":
    cap = VideoCapture(0)
    image = cap.read()


