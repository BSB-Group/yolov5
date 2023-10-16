from models.yolo import BaseModel, DetectionModel
from models.common import Classify, DetectMultiBackend


class HorizonModel(BaseModel):

    def __init__(self, model: DetectionModel,
                 nc_pitch: int = 500,
                 nc_theta: int = 500,
                 cutoff: int = 8,
                 mode: str = None):
        """
        Horizon detection model.

        Args:
            model (DetectionModel): YOLOv5 model
            nc_pitch (int, optional): number of classes for pitch classification. Defaults to 500.
            nc_theta (int, optional): number of classes for theta classification. Defaults to 500.
            cutoff (int, optional): cutoff layer for classification heads. Defaults to 8.
            mode (str, optional): mode of operation. Possible values: "horizon", "detection".
                Defaults to `None` which means both "horizon" and "detection" are enabled. 
        """
        super().__init__()

        self.nc_pitch = nc_pitch
        self.nc_theta = nc_theta
        self.cutoff = cutoff
        self.mode = mode

        # check if model is an instance of DetectMultiBackend
        if isinstance(model, DetectMultiBackend):
            self.device = model.device
            self.fp16 = model.fp16

            model = model.model  # unwrap DetectMultiBackend
            c_pitch, c_theta = self._from_detection_model(model)

            self.model = model.model
            self.stride = model.stride
            self.save = list(set(model.save + [self.cutoff]))  # avoid duplication
            self.nc = model.nc

            # add classification heads to model
            self.model.add_module(c_pitch.i, c_pitch)
            self.model.add_module(c_theta.i, c_theta)
        else:
            raise TypeError(f"Model must be an instance of {DetectMultiBackend}")

    def _from_detection_model(self, model):
        """
        Get classification heads. Similar to method found in
        `models.yolo.ClassificationModel._from_detection_model`
        """

        # get number of input channels for classification heads
        m = model.model[self.cutoff + 1]  # layer after cutoff
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module

        # define classification heads
        c_pitch = Classify(ch, self.nc_pitch).to(self.device)
        c_pitch.i, c_pitch.f, c_pitch.type = 'c_pitch', self.cutoff, 'models.common.Classify'
        c_theta = Classify(ch, self.nc_theta).to(self.device)
        c_theta.i, c_theta.f, c_theta.type = 'c_theta', self.cutoff, 'models.common.Classify'

        return c_pitch, c_theta

    def _horizon_once(self, x):
        x_pitch, x_theta = None, None
        y, dt = [], []  # outputs
        for m in self.model:
            if isinstance(m.i, int) and m.i > self.cutoff:
                continue
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if m.type == 'models.common.Classify' and 'pitch' in m.i:
                x_pitch = m(x)
            elif m.type == 'models.common.Classify' and 'theta' in m.i:
                x_theta = m(x)
            else:  # object detection flow
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output

        return (x_pitch, x_theta)

    def _detect_once(self, x):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.type == 'models.common.Classify':
                continue
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    def _forward_once(self, x):
        y, dt = [], []  # outputs
        x_pitch, x_theta = None, None
        for m in self.model:
            if m.type == 'models.common.Classify':
                # horizon flow
                xf = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                if 'pitch' in m.i:
                    x_pitch = m(xf)
                if 'theta' in m.i:
                    x_theta = m(xf)
                continue  # skip to next layer

            # object detection flow
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        return (x, x_pitch, x_theta)

    def forward(self, x):
        if self.mode == "detection":
            return self._detect_once(x)
        elif self.mode == "horizon":
            return self._horizon_once(x)
        else:
            return self._forward_once(x)


class HorizonRegModel(BaseModel):

    def __init__(self, model: DetectionModel,
                 nc_horizon: int = 2,
                 cutoff: int = 8,
                 mode: str = None):
        """
        Horizon detection model.

        Args:
            model (DetectionModel): YOLOv5 model
            nc_pitch (int, optional): number of classes for pitch classification. Defaults to 500.
            nc_theta (int, optional): number of classes for theta classification. Defaults to 500.
            cutoff (int, optional): cutoff layer for classification heads. Defaults to 8.
            mode (str, optional): mode of operation. Possible values: "horizon", "detection".
                Defaults to `None` which means both "horizon" and "detection" are enabled. 
        """
        super().__init__()

        self.nc_horizon = nc_horizon
        self.cutoff = cutoff
        self.mode = mode

        # check if model is an instance of DetectMultiBackend
        if isinstance(model, DetectMultiBackend):
            self.device = model.device
            self.fp16 = model.fp16

            model = model.model  # unwrap DetectMultiBackend
            horizon = self._from_detection_model(model)

            self.model = model.model
            self.stride = model.stride
            self.save = list(set(model.save + [self.cutoff]))  # avoid duplication
            self.nc = model.nc

            # add classification heads to model
            self.model.add_module(horizon.i, horizon)
        else:
            raise TypeError(f"Model must be an instance of {DetectMultiBackend}")

    def _from_detection_model(self, model):
        """
        Get classification heads. Similar to method found in
        `models.yolo.ClassificationModel._from_detection_model`
        """

        # get number of input channels for classification heads
        m = model.model[self.cutoff + 1]  # layer after cutoff
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module

        # define classification heads
        horizon = Classify(ch, self.nc_horizon).to(self.device)
        horizon.i, horizon.f, horizon.type = 'horizon', self.cutoff, 'models.common.Classify'

        return horizon

    def _horizon_once(self, x):
        x_horizon = None
        y, dt = [], []  # outputs
        for m in self.model:
            if isinstance(m.i, int) and m.i > self.cutoff:
                continue
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if m.type == 'models.common.Classify' and 'horizon' in m.i:
                x_horizon = m(x)
            else:  # object detection flow
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output

        return x_horizon

    def _detect_once(self, x):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.type == 'models.common.Classify':
                continue
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    def _forward_once(self, x):
        y, dt = [], []  # outputs
        x_horizon = None
        for m in self.model:
            if m.type == 'models.common.Classify' and 'horizon' in m.i:
                # horizon flow
                xf = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                x_horizon = m(xf)
                continue  # skip to next layer

            # object detection flow
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        return (x, x_horizon)

    def forward(self, x):
        if self.mode == "detection":
            return self._detect_once(x)
        elif self.mode == "horizon":
            return self._horizon_once(x)
        else:
            return self._forward_once(x)

    def print_layers(self):
        for m in self.model:
            print(m.i, m.f, m.type)
