import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from pathlib import Path
from tqdm import tqdm
import os
import sys
from line_profiler import profile

sys.setrecursionlimit(10000)


DEBUG=os.getenv('DEBUG')
AUGMENT= not DEBUG or os.getenv('AUGMENT')

# INCREMENTS = 120
# START = 0
# END = 0.99
ORIG_HEIGHT = 1848
ORIG_WIDTH = 3280
SCALE_FACTOR = 3
CURRENT_FILEPATH = Path(__file__).parent.absolute()

from cyclone_a2rl.shared.constants import APRIL_ARENA_BOUNDS, APRIL_GATE_LOCATIONS, CV_WORLD_FRAME_TO_WORLD_FRAME_MAT, EXTERIOR_COORDS_WORLD, GATE_EXTERIOR_DIMENSION
from cyclone_a2rl.shared.util import wp_inline_with_gate



def spiral_position(t, start_yaw, total_yaw, start_elev, end_elev, radius_start, radius_end, spiral_bend_fn):
    return Rotation.from_euler('z', start_yaw + t*total_yaw, degrees=True).apply(np.array([radius_start+spiral_bend_fn(t)*(radius_end-radius_start),0,0])) + np.array([0,0,start_elev + t*(end_elev-start_elev)])

gates = APRIL_GATE_LOCATIONS
arena_bounds = APRIL_ARENA_BOUNDS

path = [ # xyz, yaw
    (8, 22, 0.2, 35), # takeoff point
    (8, 22, 1.35, 35), # takeoff point
    wp_inline_with_gate(gates[0], -1), # gate 1
    (*wp_inline_with_gate(gates[0], 1)[:3], gates[1][3]), # gate 1
    wp_inline_with_gate(gates[1], -2), # gate 2
    (*wp_inline_with_gate(gates[1], 2)[:3], gates[2][3]), # gate 2
    wp_inline_with_gate(gates[2], -2), # gate 3
    (*wp_inline_with_gate(gates[2], 2)[:3], gates[3][3]), # gate 3
    wp_inline_with_gate(gates[4], -2), # DG1 top
    wp_inline_with_gate(gates[4], 1), # DG1 top
    wp_inline_with_gate(gates[5], -1), # DG2 top
    wp_inline_with_gate(gates[5], 1, True), # DG2 top
    wp_inline_with_gate(gates[6], 1, True), # DG2 bottom
    wp_inline_with_gate(gates[6], -1, True), # DG2 bottom
    wp_inline_with_gate(gates[7], 1, True), # G4
    wp_inline_with_gate(gates[7], -1, True), # G4
    wp_inline_with_gate(gates[8], 1, True), # G5
    wp_inline_with_gate(gates[8], -1, True), # G5
    wp_inline_with_gate(gates[9], 1, True), # G6
    wp_inline_with_gate(gates[9], -1, True), # G6
    wp_inline_with_gate(gates[10], 1, True), # G7
    wp_inline_with_gate(gates[10], -1, True), # G7
    wp_inline_with_gate(gates[11], -1), # G8
    wp_inline_with_gate(gates[11], 1), # G8
    (8,22,1,35.), # takeoff point
]
path = np.array(path)

# distortion_coefficients = np.zeros((1,5))
# cv2.namedWindow('img', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('img', 1600, 900)

texture = cv2.imread(f'{CURRENT_FILEPATH}/gate_texture.png')

background_textures = list(Path(f'{CURRENT_FILEPATH}/background_textures').glob('*.jpg'))
bg_textures_imgs = [cv2.imread(str(p)) for p in background_textures]

def sample_xyz_by_time(time):
    index = np.floor(time*(len(path)-1))
    wp_before = path[int(index)]
    wp_after = path[int(index+1)]
    t = (time*(len(path)-1) - index)
    wp_interp = wp_before + t*(wp_after - wp_before)
    return wp_interp

def lookahead(wp):
    return wp[:3] + Rotation.from_euler('z', wp[3], degrees=True).apply(np.array([1,0,0]))

equal_spacing = np.linspace(0,0.5,10_000)
weights = np.array([
    min(np.linalg.norm(np.array(lookahead(sample_xyz_by_time(t))) - g[:3]) for g in gates)
    for t in equal_spacing
])
weights = 1/(weights**6+1)
weights/=np.sum(weights)

# from matplotlib import pyplot as plt
# plt.plot(equal_spacing, weights)
# plt.show()


def sample_biased():
    return np.random.choice(equal_spacing, p=weights)


def sample_pose_from_flight_path(time=None):
    # the waypoints right before the gates are duplicated to make them more likely in the random sampling
    time = sample_biased()
    if time is None:
        time = sample_biased()
        # time = np.random.uniform(0, 1)
        # if np.random.uniform(0,1) > 0.7: # 70% chance to sample only the first part of the flight path
        #     time = np.sqrt(time)/5 # hack to over-sample just the initial part
    wp_interp = sample_xyz_by_time(time)
    x,y,z, yaw = wp_interp
    # translate to opencv coordinates
    x_arr = np.array([x, y, z], dtype=np.float32)
    dist_to_closest_gate = min(np.linalg.norm(np.array([x,y,z]) - g[:3]) for g in gates)
    pos_noise = np.random.normal(0,dist_to_closest_gate*np.array([0.3,0.2,0.1]),3)
    x_arr += pos_noise
    tvec = x_arr @ CV_WORLD_FRAME_TO_WORLD_FRAME_MAT
    if AUGMENT:
        cam_angle = np.random.uniform(15,45)
        euler = np.array([-cam_angle, yaw, 0])
        euler_noise = np.random.uniform(-10, 10, 3)
        euler += euler_noise
    else:
        euler = np.array([-30, yaw, 0])

    # point_cv_cam = r_cv @ point_cv_world + t_cv

    r = Rotation.from_euler('XYZ', euler, degrees=True)
    rvec = r.as_rotvec()

    return rvec, -r.apply(tvec)

camera_matrices = []
distortion_coefficients_list = []
new_cam_mats = []
inv_distort_maps_list = []

for _ in tqdm(range(20)):
    if np.random.randint(0,2) == 0:
        calib_dict = {"matrix": [[1158.580361210498,0.0,1636.6360848308107],[0.0, 1158.580361210498,917.252640686633],[0.0,0.0,1.0]],"distortion":[[0.027784572303817985,-0.26699049185215107,0.0003233929383388227,-0.0001881871645317549,-0.01224272891718848,0.3106779743035149,  -0.32681736130439565,-0.05882340985843282,0.0,0.0,0.0,0.0,0.0,0.0]]}
    else:
        calib_dict = {"matrix": [[1150.3956978402705, 0.0, 1599.8912802286425], [0.0, 1150.3956978402705, 936.5565118474698], [0.0, 0.0, 1.0]], "distortion": [[0.8552633417559007, 0.14403165519429206, 1.6870898851992677e-05, -9.163413609619273e-05, 0.002150176593346323, 1.1348031591562646, 0.32084729305960985, 0.01652388705858197, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]}
    cam_mat = np.array(calib_dict['matrix'])
    distortion_coefficients = np.array(calib_dict['distortion'][0])
    cam_mat_multipliers = np.random.uniform(0.9, 1.1, 4)
    for k, (i, j) in enumerate([(0,0), (0,2), (1,1), (1,2)]):
        cam_mat[i, j] *= cam_mat_multipliers[k]
    cam_mat[:2, :] /= SCALE_FACTOR
    camera_matrices.append(cam_mat)
    distortion_coefficients_list.append(distortion_coefficients)

    new_cam_mat, _ = cv2.getOptimalNewCameraMatrix(cam_mat, distortion_coefficients, (ORIG_WIDTH//SCALE_FACTOR, ORIG_HEIGHT//SCALE_FACTOR), 1, (ORIG_WIDTH//SCALE_FACTOR, ORIG_HEIGHT//SCALE_FACTOR))
    inv_distort_maps = cv2.initInverseRectificationMap(cam_mat, distortion_coefficients, None, new_cam_mat, (ORIG_WIDTH//SCALE_FACTOR, ORIG_HEIGHT//SCALE_FACTOR), cv2.CV_16SC2) # type: ignore

    new_cam_mats.append(new_cam_mat)
    inv_distort_maps_list.append(inv_distort_maps)

timepoint = 0
@profile
def make_datapoint():
    global timepoint
    img = np.zeros((ORIG_HEIGHT//SCALE_FACTOR, ORIG_WIDTH//SCALE_FACTOR, 3), dtype=np.uint8)
    if not DEBUG: timepoint = None
    rvec, tvec=  sample_pose_from_flight_path(timepoint)
    if DEBUG: timepoint+=1/100 # type: ignore

    intrinsics_index = np.random.randint(0, len(camera_matrices))
    cam_mat = camera_matrices[intrinsics_index]
    distortion_coefficients = distortion_coefficients_list[intrinsics_index]
    new_cam_mat = new_cam_mats[intrinsics_index]
    inv_distort_maps = inv_distort_maps_list[intrinsics_index]
    
    def draw_texture_on_image(img, texture, texture_cv_world_coords, rvec, tvec):
        corners_transformed = cv2.projectPoints(texture_cv_world_coords.astype(np.float32), rvec, tvec, new_cam_mat, np.zeros((1,5)))[0].squeeze()
        transform = cv2.getPerspectiveTransform(np.array([[0,0], [texture.shape[1], 0], [texture.shape[1], texture.shape[0]], [0, texture.shape[0]]], dtype=np.float32), corners_transformed)
        just_the_gate = cv2.warpPerspective(texture, transform, (img.shape[1], img.shape[0]))
        just_the_gate =  cv2.remap(just_the_gate, inv_distort_maps[0], inv_distort_maps[1], cv2.INTER_LINEAR)
        img[just_the_gate > 0] = just_the_gate[just_the_gate > 0]

    for arena_side_coordinates in [
        [ # back wall
            [arena_bounds[0][1], arena_bounds[1][0], arena_bounds[2][0]], 
            [arena_bounds[0][1], arena_bounds[1][1], arena_bounds[2][0]],
            [arena_bounds[0][1], arena_bounds[1][1], arena_bounds[2][1]],
            [arena_bounds[0][1], arena_bounds[1][0], arena_bounds[2][1]], 
        ],
        [ # right wall
            [arena_bounds[0][1], arena_bounds[1][0], arena_bounds[2][1]], 
            [arena_bounds[0][0], arena_bounds[1][0], arena_bounds[2][1]],
            [arena_bounds[0][0], arena_bounds[1][0], arena_bounds[2][0]],
            [arena_bounds[0][1], arena_bounds[1][0], arena_bounds[2][0]], 
        ],
        [ # left wall
            [arena_bounds[0][0], arena_bounds[1][1], arena_bounds[2][1]], 
            [arena_bounds[0][1], arena_bounds[1][1], arena_bounds[2][1]],
            [arena_bounds[0][1], arena_bounds[1][1], arena_bounds[2][0]],
            [arena_bounds[0][0], arena_bounds[1][1], arena_bounds[2][0]], 
        ],
        [ # floor
            [arena_bounds[0][0], arena_bounds[1][0], arena_bounds[2][0]], 
            [arena_bounds[0][1], arena_bounds[1][0], arena_bounds[2][0]],
            [arena_bounds[0][1], arena_bounds[1][1], arena_bounds[2][0]],
            [arena_bounds[0][0], arena_bounds[1][1], arena_bounds[2][0]], 
        ],
        [ # ceiling
            [arena_bounds[0][0], arena_bounds[1][0], arena_bounds[2][1]], 
            [arena_bounds[0][1], arena_bounds[1][0], arena_bounds[2][1]],
            [arena_bounds[0][1], arena_bounds[1][1], arena_bounds[2][1]],
            [arena_bounds[0][0], arena_bounds[1][1], arena_bounds[2][1]], 
        ],
    ]:
        random_idx = np.random.randint(len(bg_textures_imgs))
        bg_texture = bg_textures_imgs[random_idx]
        draw_texture_on_image(img, bg_texture, np.array(arena_side_coordinates) @ CV_WORLD_FRAME_TO_WORLD_FRAME_MAT, rvec, tvec)

    # this computation is duplicated but IDC because it's cheap
    def sort_key(gate_loc):
        gate_pos = gate_loc[:3]
        gate_yaw = gate_loc[3]
        gate_corners_3d = (Rotation.from_euler('z', gate_yaw, degrees=True).apply(EXTERIOR_COORDS_WORLD) + gate_pos) @ CV_WORLD_FRAME_TO_WORLD_FRAME_MAT
        rvec_rot = Rotation.from_rotvec(rvec)
        gate_corners_cam = rvec_rot.apply(gate_corners_3d) + tvec
        min_z = gate_corners_cam[:,2].min()
        return -min_z

    label_lines = []
    for gate_loc in sorted(gates, key=sort_key):
        gate_pos = gate_loc[:3]
        gate_yaw = gate_loc[3]
        gate_corners_3d = (Rotation.from_euler('z', gate_yaw, degrees=True).apply(EXTERIOR_COORDS_WORLD) + gate_pos) @ CV_WORLD_FRAME_TO_WORLD_FRAME_MAT
        rvec_rot = Rotation.from_rotvec(rvec)
        gate_corners_cam = rvec_rot.apply(gate_corners_3d) + tvec
        min_z = gate_corners_cam[:,2].min()
        if min_z < 0:
            continue
        do_mirror_gate = gate_corners_cam[0][0] > gate_corners_cam[1][0]
        if do_mirror_gate:
            continue

        points = []
        for i in range(10):
            for j in range(10):
                if i > 2 and i <= 6 and j > 2 and j <= 6:
                    continue
                points.append([i/9, j/9, 0])
        colors_for_visualization = [
            (int(x*255), int(y*255), int(z*255)) for x,y,z in points
        ]
        points = GATE_EXTERIOR_DIMENSION*(np.array([0.5,0.5,0]) - np.array(points))
        points = Rotation.from_euler('y', -gate_yaw, degrees=True).apply(points) + gate_pos@CV_WORLD_FRAME_TO_WORLD_FRAME_MAT
        grid_proj_inaccurate = cv2.projectPoints(points, rvec, tvec, cam_mat, distortion_coefficients)[0].squeeze()

        grid_proj_undistored = cv2.projectPoints(points, rvec, tvec, new_cam_mat, np.zeros((1,5)))[0].squeeze()
        def is_visible(point):
            return point[0] >= 0 and point[0] < img.shape[1] and point[1] >= 0 and point[1] < img.shape[0]
        
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
                if x<0 or y<0 or x>=img.shape[1] or y>=img.shape[0]:
                    return float('inf')
                diff = np.linalg.norm(inv_distort_maps[0][y,x] - p)
                return diff

            return dfs(x,y)


        grid_proj = np.array([
            find_closest(i) if is_visible(p) and is_visible(grid_proj_inaccurate[i]) else (-1,-1)
            for i,p in enumerate(grid_proj_undistored)
        ])

        is_visible_mask = np.array([int(is_visible(p)) for i,p in enumerate(grid_proj)])
        if is_visible_mask.sum() == 0:
            continue



        draw_texture_on_image(img, texture, gate_corners_3d, rvec, tvec)
        bbox = cv2.boundingRect(grid_proj.astype(np.float32))
        bbox_xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        bbox_xyxy = np.clip(bbox_xyxy, 0, [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        centered_bbox = np.array([(bbox_xyxy[0] + bbox_xyxy[2])/2, (bbox_xyxy[1] + bbox_xyxy[3])/2, bbox_xyxy[2] - bbox_xyxy[0], bbox_xyxy[3] - bbox_xyxy[1]])
        label_start = f'0 {centered_bbox[0]/img.shape[1]} {centered_bbox[1]/img.shape[0]} {centered_bbox[2]/img.shape[1]} {centered_bbox[3]/img.shape[0]}'
        def point_to_label_str(point,i):
            is_point_visible = is_visible(point)
            if not is_point_visible: # labels are "corrupt" otherwise even though we kinda do want these to be predicted outside the image
                return f'0.0 0.0 0'
            return f'{point[0]/img.shape[1]} {point[1]/img.shape[0]} {0 if not is_point_visible else 2}'
        # if all points are out of bounds, don't append
        labels_full_str = ' '.join([label_start, *[point_to_label_str(point,i) for i,point in enumerate(grid_proj)]])
        label_lines.append(labels_full_str)

        # visualize
        if DEBUG:
            for point, color in zip(grid_proj, colors_for_visualization):
                if not is_visible(point): continue
                cv2.circle(img, tuple(point.astype(int)), 5, color, -1)
    if AUGMENT:
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
                cv2.rectangle(img_copy, [*tuple(((center - size/2)*img.shape[:2]).astype(int)), *tuple(((center + size/2)*img.shape[:2]).astype(int),)], tuple(color), -1)
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
    if DEBUG:
        cv2.imshow('img', img)
        cv2.waitKey(0)

    # make grid of 9x9 grid of points, but excluding a square from (2/9, 2/9 to 7/9, 7/9)
    return img, '\n'.join(label_lines)

DATASET_SIZE = 100 if DEBUG else 10_000
VAL_SPLIT = 0.1
TRAIN_SIZE = int(DATASET_SIZE * (1 - VAL_SPLIT))
VAL_SIZE = DATASET_SIZE - TRAIN_SIZE
imgs_dir = Path('images')
labels_dir = Path('labels')
imgs_dir.mkdir(exist_ok=True)
labels_dir.mkdir(exist_ok=True)
train_images_dir = imgs_dir / 'train'
train_labels_dir = labels_dir / 'train'
val_images_dir = imgs_dir / 'val'
val_labels_dir = labels_dir / 'val'

train_images_dir.mkdir(exist_ok=True)
train_labels_dir.mkdir(exist_ok=True)
val_images_dir.mkdir(exist_ok=True)
val_labels_dir.mkdir(exist_ok=True)

for i in tqdm(range(TRAIN_SIZE)):
    img, labels_full_str = make_datapoint()
    cv2.imwrite(str(train_images_dir / f'{i}.png'), img)
    with open(train_labels_dir / f'{i}.txt', 'w') as f:
        f.write(labels_full_str)

for i in tqdm(range(VAL_SIZE)):
    img, labels_full_str = make_datapoint()
    cv2.imwrite(str(val_images_dir / f'{i}.png'), img)
    with open(val_labels_dir / f'{i}.txt', 'w') as f:
        f.write(labels_full_str)

# # draw grid
# for i in range(grid_proj.shape[0]):
#     cv2.circle(img, tuple(grid_proj[i].astype(int)), 3, (0, 255, 0), -1)

# cv2.imshow('img', img)
# cv2.waitKey(0)