import numpy as np
import matplotlib.pyplot as plt
import os

def get_xycoords_for_plotting(vframe, number_of_markers_or_joints, _3d):
    xcoords = []
    ycoords = []
    zcoords = []
    for n, i in enumerate(vframe):
        xcol = []
        ycol = []
        zcol = []
        for j in range(0,number_of_markers_or_joints):
            xcol.append(i[j][0])
            ycol.append(i[j][1])
            if _3d == '3d' :
                zcol.append(i[j][2])
        xcoords.append(xcol)
        ycoords.append(ycol)
        if _3d == '3d' :
            zcoords.append(zcol)
    return xcoords, ycoords, zcoords



















def get_joint_connections():
    connections = [
        (0, 16), (0, 21), (0, 1),  # Hips to legs
        (16, 17), (17, 18), (18, 19), (19, 20), # Left leg
        (21, 22), (22, 23), (23, 24), (24, 25), # Right leg
        (1, 2), (2, 3), (3, 4), (4, 5), # Spine to Head
        (3, 6), (6, 7), (7, 8), (8, 9), (9, 10),# Left arm
        (3, 11), (11, 12), (12, 13), (13, 14), (14, 15),# Right arm
    ]
    return connections

def plot_frame(vframes, xcoords, ycoords, zcoords, idx, connections, marker):
    plt.ion()
    print(vframes.shape[0])
    for i in range(0,vframes.shape[0],1000):
        print(i)   
        plt.clf()
        plt.scatter(xcoords[idx], ycoords[idx], c='red', marker='o', s=50)
        # Plot bones
        if marker != 'marker' :
            for start_idx, end_idx in connections:
                start_point = vframes[idx][start_idx]
                end_point = vframes[idx][end_idx]
                plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'b-')
        plt.title('2D Body Pose Estimation')
        plt.xlim(plt.xlim()[::-1])
        plt.ylim(plt.ylim()[::-1])
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box') # Maintain aspect ratio
        plt.pause(0.5)  # Pause to create animation effect
    plt.ioff()
    plt.show()

def load_video_frames(directory_path):
    
    vframes = []
    ffiles = []
    vextns = ['.npy']
    
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return []

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            ffile, extension = os.path.splitext(filename)
            if extension.lower() in vextns:
                vframe = np.load(file_path)
                vframes.append(vframe)
                ffiles.append(ffile)
                
    return vframes, ffiles


def main():
    # Read all data files
    ex1_vframes, ex1_ffiles = load_video_frames('2d_joints/Ex1')
    ex2_vframes, ex2_ffiles = load_video_frames('2d_joints/Ex2')
    ex3_vframes, ex3_ffiles = load_video_frames('2d_joints/Ex3')
    ex4_vframes, ex4_ffiles = load_video_frames('2d_joints/Ex4')
    ex5_vframes, ex5_ffiles = load_video_frames('2d_joints/Ex5')
    ex6_vframes, ex6_ffiles = load_video_frames('2d_joints/Ex6')
    ex1_mkr_vframes, ex1_mkr_ffiles = load_video_frames('2d_markers/Ex1')
    ex2_mkr_vframes, ex2_mkr_ffiles = load_video_frames('2d_markers/Ex2')
    ex3_mkr_vframes, ex3_mkr_ffiles = load_video_frames('2d_markers/Ex3')
    ex4_mkr_vframes, ex4_mkr_ffiles = load_video_frames('2d_markers/Ex4')
    ex5_mkr_vframes, ex5_mkr_ffiles = load_video_frames('2d_markers/Ex5')
    ex6_mkr_vframes, ex6_mkr_ffiles = load_video_frames('2d_markers/Ex6')
    ex1_3d_vframes, ex1_3d_ffiles = load_video_frames('3d_joints/Ex1')
    ex2_3d_vframes, ex2_3d_ffiles = load_video_frames('3d_joints/Ex2')
    ex3_3d_vframes, ex3_3d_ffiles = load_video_frames('3d_joints/Ex3')
    ex4_3d_vframes, ex4_3d_ffiles = load_video_frames('3d_joints/Ex4')
    ex5_3d_vframes, ex5_3d_ffiles = load_video_frames('3d_joints/Ex5')
    ex6_3d_vframes, ex6_3d_ffiles = load_video_frames('3d_joints/Ex6')
    ex1_mkr3d_vframes, ex1_mkr3d_ffiles = load_video_frames('3d_markers/Ex1')
    ex2_mkr3d_vframes, ex2_mkr3d_ffiles = load_video_frames('3d_markers/Ex2')
    ex3_mkr3d_vframes, ex3_mkr3d_ffiles = load_video_frames('3d_markers/Ex3')
    ex4_mkr3d_vframes, ex4_mkr3d_ffiles = load_video_frames('3d_markers/Ex4')
    ex5_mkr3d_vframes, ex5_mkr3d_ffiles = load_video_frames('3d_markers/Ex5')
    ex6_mkr3d_vframes, ex6_mkr3d_ffiles = load_video_frames('3d_markers/Ex6')

    for j in range(0, len(ex1_vframes),50):
        vframe = ex1_vframes[j]
        xcoords, ycoords, zcoords = get_xycoords_for_plotting(vframe, 26, '2d')
        connections = get_joint_connections()
        for idx in range(0, vframe.shape[0], 10):
            plot_frame(vframe, xcoords, ycoords, zcoords, idx, connections, 'joint')

if __name__ == "__main__":
    main()