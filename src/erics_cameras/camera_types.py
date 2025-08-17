from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Generator, Generic, NamedTuple, TypeVar, Union
import cv2
import numpy as np
from enum import Enum
from scipy.spatial.transform import Rotation


integer = Union[
    int,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
]
real = Union[float, np.float16, np.float32, np.float64]
number = Union[integer, real]

# Can't use np.uint16 because torch doesn't support it. We're good as long as we don't have a gigapixel camera.
img_coord_t = np.int16


class ImageDimension(Enum):
    HEIGHT = "h"
    WIDTH = "w"
    CHANNELS = "c"


HEIGHT = ImageDimension.HEIGHT
WIDTH = ImageDimension.WIDTH
CHANNELS = ImageDimension.CHANNELS


class ImageDimensionsOrder(NamedTuple):
    first_dim: ImageDimension
    second_dim: ImageDimension
    third_dim: ImageDimension


# Can support other dimension orders if necessary
HWC = ImageDimensionsOrder(HEIGHT, WIDTH, CHANNELS)
CHW = ImageDimensionsOrder(CHANNELS, HEIGHT, WIDTH)
_VALID_DIM_ORDERS = {HWC, CHW}

_UnderlyingImageT = TypeVar("_UnderlyingImageT", np.ndarray, np.ndarray)

_id = 0


class Image(Generic[_UnderlyingImageT]):
    """
    Wraps a numpy array or torch tensor representing an image.
    Contains information about the dimension order of the underlying array, e.g., (height, width, channels) or (channels, height, width).

    Except for passing data to predictors, you should interface through it directly instead of accessing _array.
    NOTE: Add methods to interface with it if necessary.

    Args:
        array (np.ndarray | torch.Tensor): The underlying array
        dim_order (ImageDimensionsOrder): The dimension order of the underlying array

    Examples:
        image_hwc[np.ndarray] = Image(np.zeros((20, 20, 3)), HWC)

        image_chw[torch.Tensor] = Image(torch.zeros((3, 20, 20)), CHW)
    """

    def __init__(self, array: _UnderlyingImageT, dim_order: ImageDimensionsOrder = HWC):
        if array is None:
            raise TypeError(
                "array cannot be None. Please provide a valid numpy array."
            )
        
        if not isinstance(array, np.ndarray):
            raise TypeError(
                f"array must be a numpy array. Got {type(array)}"
            )

        if len(array.shape) != 3:
            raise ValueError("array must have 3 axes, got shape " + str(array.shape))

        if dim_order not in _VALID_DIM_ORDERS:
            raise ValueError("dim_order must be one of " + str(_VALID_DIM_ORDERS))

        self._dim_order = dim_order

        channels_index = self._dim_order.index(CHANNELS)
        if array.shape[channels_index] != 3:
            raise ValueError(
                f"Image array must have 3 channels, got {array.shape[channels_index]} for shape {array.shape}"
            )

        self._array: _UnderlyingImageT = array
        global _id
        self._id = _id
        _id += 1

    def __getitem__(self, key):
        return self._array[key]

    def __setitem__(
        self,
        key,
        value: Union[
            np.ndarray, "Image", int, float, np.number
        ],
    ):
        if isinstance(value, Image):
            value = value._array

        # I'm not sure why this thrown a fit in VS Code, but it work. Trust.
        self._array[key] = value  # type: ignore

    def __eq__(self, other: object) -> bool:
        """
        Checks whether two images are equal, including whether they have the same dimension order.
        """
        return (
            isinstance(other, Image)
            and self._dim_order == other._dim_order
            and (self._array == other._array).all()
        )

    def __repr__(self):
        return f"Image({self._array}, {self._dim_order})"

    def __mul__(self, other: number | _UnderlyingImageT) -> "Image":
        """
        Multiplies the underlying array by a scalar or another array/tensor.
        """
        return Image(self._array * other, self._dim_order)

    def get_array(self) -> _UnderlyingImageT:
        return self._array

    @property
    def shape(self):
        return self._array.shape

    @property
    def dim_order(self):
        return self._dim_order

    @property
    def height(self):
        return self._array.shape[self._dim_order.index(HEIGHT)]

    @property
    def width(self):
        return self._array.shape[self._dim_order.index(WIDTH)]

    @property
    def channels(self):
        return self._array.shape[self._dim_order.index(CHANNELS)]

    @staticmethod
    def from_file(
        fp: str,
        dim_order: ImageDimensionsOrder = HWC,
        array_type: type[np.ndarray] = np.ndarray,
        dtype: type[integer] = np.uint8,
    ) -> "Image[np.ndarray]":
        """
        Reads an image from a file. Uses cv2.imread internally, so the image will be in BGR format.

        Args:
            fp (str): The file path
            dim_order (ImageDimensionsOrder, optional): The desired dimension order of the underlying array. Defaults to HWC, cv2's default.
            array_type (type[np.ndarray | torch.Tensor], optional): The type of the underlying array. Defaults to np.ndarray.
            dtype (type[integer], optional): The type of the underlying array's elements. Defaults to np.uint8.

        Returns:
            Image: The image
        """
        if array_type == np.ndarray:
            array = cv2.imread(fp).astype(dtype)
            img = Image(array, HWC)
            if dim_order != HWC:
                img.change_dim_order(dim_order)

            return img

        else:
            raise TypeError("array_type must be np.ndarray or torch.Tensor")

    def save(self, fp: os.PathLike | str) -> None:
        """
        Saves the image to a file. Uses cv2.imwrite internally.


        Args:
            fp (str): The file path
        """
        np_array = np.array(self._array)

        # Tranpose if necessary.
        if self._dim_order == CHW:
            np_array = np_array.transpose(1, 2, 0)

        cv2.imwrite(str(fp) + f"/{self._id}.jpg", np_array)

    def change_dim_order(self, target_dim_order: ImageDimensionsOrder) -> None:
        """
        Use transpose to change the order of the dimensions in-place. This does NOT copy the underlying array.
        Changes the dim_order accordingly.
        """
        transposition_indices = (
            self._dim_order.index(target_dim_order.first_dim),
            self._dim_order.index(target_dim_order.second_dim),
            self._dim_order.index(target_dim_order.third_dim),
        )

        if isinstance(self._array, np.ndarray):
            self._array = self._array.transpose(transposition_indices)
        else:
            TypeError("Inner array must be a numpy array or torch tensor")

        self._dim_order = target_dim_order

