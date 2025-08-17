import numpy as np
import cv2
import sys

sys.setrecursionlimit(10000)

class Billboard:
    def __init__(self, texture: cv2.Mat, cv_world_coords: np.ndarray):
        self.texture = texture
        self.cv_world_coords = cv_world_coords

    @staticmethod
    def from_pose_and_size(texture, tvec, rvec, size):
        '''
        tvec is 3x1
        rvec is 3x1
        size is (width, height)
        When tvec is 0 and rvec is 0, the billboard is centered at the origin, in-line with the image plane (i.e. x-axis facing right, y-axis facing down)
        '''
        # figure out corners in world frame
        world_coords_before_transform = np.array([
            [-size[0]/2, -size[1]/2, 0],
            [size[0]/2, -size[1]/2, 0],
            [size[0]/2, size[1]/2, 0],
            [-size[0]/2, size[1]/2, 0]
        ], dtype=np.float32)
        world_coords_after_transform =  world_coords_before_transform @ cv2.Rodrigues(rvec)[0].T + tvec.flatten()
        return Billboard(texture, world_coords_after_transform)


class OpenCVRenderer:
    def __init__(self, cam_matrix = None, distortion_coeffs = None):
        '''
        cam_matrix is 3x3 and optional, it can be provided at render-time
        distortion_coeffs is 4x1 for fisheye cameras and optional, it can be provided at render-time
        '''
        self.billboards = []
        self.cam_matrix = cam_matrix
        self.distortion_coeffs = distortion_coeffs

    def render_image(self, resolution: tuple[int,int], rvec: np.ndarray, tvec: np.ndarray, cam_matrix=None, distortion_coeffs=None):
        '''
        resolution is (width, height)
        rvec and tvec need to be shaped (3,1) and float32 dtype
        '''
        assert rvec.shape == (3,1), f'rvec must be shaped (3,1), got {rvec.shape}'
        assert tvec.shape == (3,1), f'tvec must be shaped (3,1), got {tvec.shape}'

        img = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)

        if self.cam_matrix is not None:
            cam_matrix = self.cam_matrix
        if self.distortion_coeffs is not None:
            distortion_coeffs = self.distortion_coeffs

        if cam_matrix is None:
            raise ValueError('cam_matrix is required either at init or at render-time')
        if distortion_coeffs is None:
            raise ValueError('distortion_coeffs is required either at init or at render-time')

        # For fisheye cameras, we use fisheye-specific undistortion
        # Create undistortion maps using fisheye functions
        new_K = cam_matrix.copy()
        
        # Import the utility function
        from fisheye_utils import create_fisheye_inverse_maps
        
        # Generate inverse maps
        inv_distort_maps = create_fisheye_inverse_maps(cam_matrix, distortion_coeffs, resolution)

        for billboard in self.billboards:
            self._draw_texture_on_image(img, billboard, rvec.astype(np.float32), tvec.astype(np.float32), new_K, inv_distort_maps)

        return img

    def add_billboard(self, billboard_texture, billboard_cv_world_coords):
        self.billboards.append(Billboard(billboard_texture, billboard_cv_world_coords))
    
    def add_billboard_from_pose_and_size(self, texture, rvec, tvec, size):
        billboard = Billboard.from_pose_and_size(texture, tvec.astype(np.float32), rvec.astype(np.float32), size)
        self.billboards.append(billboard)

    def _draw_texture_on_image(self, img, billboard: Billboard, rvec, tvec, new_cam_mat, inv_distort_maps):
        # make black pixels have value 1 so they don't get masked out
        billboard_texture_copy = billboard.texture.copy()
        billboard_texture_copy[billboard_texture_copy == 0] = 1
        corners_transformed = cv2.fisheye.projectPoints(billboard.cv_world_coords.astype(np.float32).reshape(-1, 1, 3), rvec, tvec, new_cam_mat, np.zeros((4, 1), dtype=np.float32))[0].squeeze()
        transform = cv2.getPerspectiveTransform(np.array([[0,0], [billboard_texture_copy.shape[1], 0], [billboard_texture_copy.shape[1], billboard_texture_copy.shape[0]], [0, billboard_texture_copy.shape[0]]], dtype=np.float32), corners_transformed)
        just_the_gate = cv2.warpPerspective(billboard_texture_copy, transform, (img.shape[1], img.shape[0]))
        just_the_gate =  cv2.remap(just_the_gate, inv_distort_maps[0], inv_distort_maps[1], cv2.INTER_LINEAR)
        img[just_the_gate > 0] = just_the_gate[just_the_gate > 0]

    def get_keypoint_labels(self, points_3d, rvec, tvec, resolution, inv_distort_maps, img_to_render_on=None):
        # For fisheye cameras, we need to use fisheye projection
        # First project points using the fisheye camera model
        grid_proj_inaccurate = cv2.fisheye.projectPoints(
            points_3d.reshape(-1, 1, 3), 
            rvec.reshape(3, 1), 
            tvec.reshape(3, 1), 
            self.cam_matrix, 
            self.distortion_coeffs
        )[0].squeeze()

        # Project without distortion for reference
        grid_proj_undistored = cv2.fisheye.projectPoints(
            points_3d.reshape(-1, 1, 3), 
            rvec.reshape(3, 1), 
            tvec.reshape(3, 1), 
            self.cam_matrix, 
            np.zeros((4, 1), dtype=self.cam_matrix.dtype)  # No distortion for reference, same dtype as cam_matrix
        )[0].squeeze()
        
        def is_visible(point):
            return point[0] >= 0 and point[0] < resolution[0] and point[1] >= 0 and point[1] < resolution[1]
        
        def find_closest(i):
            p = grid_proj_undistored[i]
            inaccurate = grid_proj_inaccurate[i].astype(np.uint32)
            x,y = int(inaccurate[0]), int(inaccurate[1])
            def dfs(x,y):
                neighbors = [
                    (cost(x+1,y), (x+1, y)),
                    (cost(x-1,y), (x-1, y)),
                    (cost(x,y+1), (x, y+1)),
                    (cost(x,y-1), (x, y-1))
                ]
                best = min(neighbors)
                if cost(x,y) <= best[0]:
                    return x,y
                else:
                    return dfs(best[1][0], best[1][1])
                
            def cost(x,y):
                if x<0 or y<0 or x>=resolution[0] or y>=resolution[1]:
                    return float('inf')
                diff = np.linalg.norm(inv_distort_maps[0][y,x] - p)
                return diff

            return dfs(x,y)
        

        # grid_proj = np.array([
        #     find_closest(i) if is_visible(p) and is_visible(grid_proj_inaccurate[i]) else (-1,-1)
        #     for i,p in enumerate(grid_proj_undistored)
        # ])

        grid_proj = grid_proj_inaccurate

        is_visible_mask = np.array([int(is_visible(p)) for i,p in enumerate(grid_proj)])
        if is_visible_mask.sum() == 0:
            return ''

        if img_to_render_on is not None:
            for i,p in enumerate(grid_proj):
                if is_visible(p):
                    cv2.circle(img_to_render_on, tuple(p.astype(int)), 3, (0, 255, 0), -1)

        bbox = cv2.boundingRect(grid_proj.astype(np.float32))
        bbox_xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        bbox_xyxy = np.clip(bbox_xyxy, 0, [resolution[0], resolution[1], resolution[0], resolution[1]])
        centered_bbox = np.array([(bbox_xyxy[0] + bbox_xyxy[2])/2, (bbox_xyxy[1] + bbox_xyxy[3])/2, bbox_xyxy[2] - bbox_xyxy[0], bbox_xyxy[3] - bbox_xyxy[1]])
        label_start = f'0 {centered_bbox[0]/resolution[0]} {centered_bbox[1]/resolution[1]} {centered_bbox[2]/resolution[0]} {centered_bbox[3]/resolution[1]}'
        def point_to_label_str(point,i):
            is_point_visible = is_visible(point)
            if not is_point_visible: # labels are "corrupt" otherwise even though we kinda do want these to be predicted outside the image
                return '0.0 0.0 0'
            return f'{point[0]/resolution[0]} {point[1]/resolution[1]} {0 if not is_point_visible else 2}'
        # if all points are out of bounds, don't append
        labels_full_str = ' '.join([label_start, *[point_to_label_str(point,i) for i,point in enumerate(grid_proj)]])
        return labels_full_str

    def augment_image(self, img: np.ndarray):
        img_copy = np.zeros_like(img)
        # add random shapes with 0-80% opacity
        n_shapes = np.random.randint(2, 10)
        # n_shapes = 10
        for _ in range(n_shapes):
            shape = np.random.choice(['circle', 'rectangle'])
            color = tuple(map(int,np.random.randint(0, 255, 3)))
            if shape == 'circle':
                center = np.random.uniform(0, 1, 2)
                radius = np.random.uniform(0.01, 0.3)
                cv2.circle(img_copy, tuple((center*img.shape[:2]).astype(int)), int(radius*img.shape[1]), tuple(color), -1)
            elif shape == 'rectangle':
                center = np.random.uniform(0, 1, 2)
                size = np.random.uniform(0.1, 0.3, 2)
                pt1 = ((center - size/2) * img.shape[:2]).astype(int)
                pt2 = ((center + size/2) * img.shape[:2]).astype(int)
                cv2.rectangle(img_copy, tuple(pt1), tuple(pt2), tuple(color), -1)
        if np.random.uniform(0,1) < 0.1: # add banding
            color = (204,2,187)
            start = 0
            while start<img_copy.shape[0]:
                gap = int(np.random.uniform(0.1,0.7)*img_copy.shape[0])
                start+=gap
                if start>=img_copy.shape[1]:
                    break
                height = int(np.random.uniform(0.1,0.6)*img_copy.shape[0])
                cv2.rectangle(img_copy, (0, start), (img_copy.shape[1]-1, start+height), color, -1)
                start+=height

        opacity = np.random.uniform(0, 0.8)
        img = cv2.addWeighted(img, 1-opacity, img_copy, opacity, 0)
            # blur
        kernel_size = np.random.choice([3,5,7,9,11,13,15,17,19,21])
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        return img

def main():
    renderer = OpenCVRenderer()
    # create example world with a box that is red, blue, and green 
    box_red_texture = np.zeros((100,100,3), dtype=np.uint8)
    box_red_texture[:,:,0] = 255
    box_blue_texture = np.zeros((100,100,3), dtype=np.uint8)
    box_blue_texture[:,:,1] = 255
    box_green_texture = np.zeros((100,100,3), dtype=np.uint8)
    box_green_texture[:,:,2] = 255
    # renderer.add_billboard_from_pose_and_size(box_red_texture, np.array([0,0,0]), np.array([0,0,-2]), (10,10))
    renderer.add_billboard(box_red_texture, np.array([[-1, 1, 2], [1, 1, 2], [1, -1, 2], [-1, -1, 2]]))
    renderer.add_billboard_from_pose_and_size(box_blue_texture, np.array([0,0,0]), np.array([0,0,-2]), (0.5,0.5))
    # renderer.add_billboard_from_pose_and_size(box_green_texture, np.array([0,0,0]), np.array([0,0,-2]), (10,10))
    cam_position = np.array([0,0,0]).reshape(3,1)
    cam_orientation = np.array([0,0,0]).reshape(3,1)
    res = (1000,1000)
    cam_matrix = np.array([[0.2*res[0],0,0.5*res[0]], [0,0.2*res[0],0.5*res[1]], [0,0,1]], dtype=np.float32)
    # Use fisheye distortion coefficients (4x1)
    distortion_coeffs = np.array([-0.3, 0.2, 0.0, 0.0], dtype=np.float32)
    img = renderer.render_image(res, cam_position, cam_orientation, cam_matrix, distortion_coeffs)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()