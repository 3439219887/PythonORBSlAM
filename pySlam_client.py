# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 11:33:04 2021

@author: Yitao Yu
"""
from open3d import *
import open3d as o3d
from pySlam import Frame
import numpy as np
import cv2
import pickle

def Visualize(cap, params):
    Visualize.cap = cap
    Visualize.params = params
    
    def update(vis):
        cap = Visualize.cap
        params = Visualize.params
        
        if cap.isOpened():
            ret, image = cap.read()
            if ret:
                frame = Frame(image)
                frame = frame.execute(params)
                # Visualizing
                    # Visualizing feature points
                points = frame.points4d
                
                if not (points is None):
                    print("features:{}".format(len(points)))
                    points = np.array(points)[:,:3]
                    pc = geometry.PointCloud()
                    pc.points = utility.Vector3dVector(points)
                    pc.paint_uniform_color([0.0, 1.0, 0.0]) # green
                    vis.add_geometry(pc)
                
                lastpose = frame.last_pose
                
                currpose = frame.curr_pose
                
                if (lastpose is None):
                    lastpose = currpose
                
                botloc = np.array(currpose[:3,3].T)
                botpc = geometry.PointCloud()
                botpc.points = utility.Vector3dVector(np.array(botloc)+0.2*np.random.rand(3, 3)) # Covenience for visualization
                botpc.paint_uniform_color([1.0, 0, 0.0]) # blue
                vis.add_geometry(botpc)
                
                ctr = vis.get_view_control()
                camera = ctr.convert_to_pinhole_camera_parameters()
                
                camera.extrinsic = lastpose # extrinsics param(pose)
                
    #            width = int(params["K"][0][2] * 2)
    #            height = int(params["K"][1][2] * 2)
    #            camera.intrinsic.set_intrinsics(width = width, height= height, fx =params["K"][0][0] ,fy =params["K"][0][1], cx = 1.0,cy =1.0) # intrinsic param(K)
                
                camera = ctr.convert_from_pinhole_camera_parameters(camera) # adjusting camera pose
                
                cv2.imshow("vid",frame.image)
                key = cv2.waitKey(1)
            
            else:
                ctr = vis.get_view_control()
                ctr.rotate(10.0, 0.0)
                cap.release()
                cv2.destroyAllWindows()
        
        return False
    
    def save():
        # save the point clouds as a file
        pass
    
    vis = o3d.visualization.Visualizer()
    
    vis.register_animation_callback(update)
    
    vis.create_window(width = params["width"], height = params["height"])
    
    vis.run()
    vis.destroy_window()
    cap.release()
    cv2.destroyAllWindows()
    save()
    pass

def VisualizeAllAtOnce(cap,params):
    Visualize.cap = cap
    Visualize.params = params

    def loop():
        filename = 'kps.pkl'
        outfile = open(filename,'wb')

        cap = Visualize.cap
        params = Visualize.params
        keypoints = []
        poses = []
        while cap.isOpened():
            ret, image = cap.read()
            if ret:
                frame = Frame(image)
                frame = frame.execute(params)
                # Visualizing
                    # Visualizing feature points
                points = frame.points4d
                
                if not (points is None):
                    points = np.array(points)[0:5,:3]
                    pickle.dump(points,outfile)
                    for p in points:
                        keypoints.append(p)
                
                lastpose = frame.last_pose
                
                currpose = frame.curr_pose
                
                if (lastpose is None):
                    lastpose = currpose
                
                botloc = np.array(currpose[:3,3].T)
                print(botloc)
                poses.append(botloc)
            else:
                break
        
        cap.release()
        outfile.close()
        print("Processing")
        print(len(keypoints))
        print(len(poses))
        
        keypoints = np.array(keypoints)
        pc = geometry.PointCloud()
        pc.points = utility.Vector3dVector(keypoints)
        pc.paint_uniform_color([0.0, 1.0, 0.0])
#        
        poses = np.array(poses)
        posespc = geometry.PointCloud()
        posespc.points = utility.Vector3dVector(poses)
        posespc.paint_uniform_color([1.0, 0.0, 0.0])
        
        return pc,posespc
    
    def save(pc,posespc):
        # save the point clouds as a file
        o3d.io.write_point_cloud('./pc.ply',pc)
        o3d.io.write_point_cloud('./poses.ply',posespc)
        pass
    
    pc,posespc = loop()
    vis = o3d.visualization.Visualizer()
        
    vis.create_window(width = params["width"], height = params["height"])
    
    vis.add_geometry(pc)
    vis.add_geometry(posespc)
    
    save(pc,posespc)

    vis.run()
    vis.destroy_window()
    pass    

if __name__ == "__main__":
    # camera intrinsics
    W, H = 960, 540
    F = 270
    K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
    
    from params import paramsdic
    
    
    cap = cv2.VideoCapture("./road.mp4")
    
    Visualize(cap, paramsdic)