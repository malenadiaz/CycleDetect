import cv2
import albumentations as A

class HorizontalFlipKeepIndexOrder(A.HorizontalFlip):
    """Flip the input horizontally but keypoint the same keypoints indices."""

    def apply_to_keypoints(self, keypoints, **params):
        flipped_keypoints = [self.apply_to_keypoint(tuple(keypoint[:4]), **params) + tuple(keypoint[4:]) for keypoint in keypoints]
        flipped_keypoints.reverse()
        return flipped_keypoints

def load_transform() ->  A.core.composition.Compose:

    input_transformer =  A.Compose([
        A.RandomBrightnessContrast(brightness_limit=1, contrast_limit=1),
        A.Flip(0.5),
        # A.RandomRotate90(0.5),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

    return input_transformer


if __name__ == '__main__':

    input_size = 224
    augmentation_type = "2chkeep"
    input_transform = None
    input_transform = load_transform(augmentation_type=augmentation_type, augmentation_probability=1.0, input_size=input_size)
