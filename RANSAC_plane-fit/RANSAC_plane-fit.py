import open3d as o3d
from mpl_toolkits import mplot3d
import numpy as np
import math
import random
import matplotlib.pyplot as plt

class fit_plane():

    # define the point cload data array and no. of iterations
    def __init__(self,pcd):
        self.pcd = pcd
        self.N = 500

    # ransac method
    def ransac(self):
        max_inlier = 0
        high_score_coeff = []
        final_inliers = []
        #ransac loop
        for i in range(self.N):
            rand_points = np.zeros((3,3))

            # take 3 random points from the point cloud data 
            for j in range(3):
                random_init = random.randint(0,len(self.pcd)-1)
                rand_points[j]=self.pcd[random_init] 
            
            # calculate the plane coefficients
            a,b,c,d = self.plane_equa_coefficients(rand_points)
            inlier_points,plane_coeff = self.inliers(a,b,c,d)

            if len(inlier_points) > len(final_inliers):
                final_inliers = inlier_points
                high_score_coeff = plane_coeff 
        
        # final number of inliers
        final_inliers = np.array(final_inliers)
        print(final_inliers)
        #print(high_score_coeff)
        return final_inliers

    #method to calculate coefficients of plane equation
    def plane_equa_coefficients(self,rand_points):
        #compute A = point1 - point0
        a1 =  rand_points[1][0] - rand_points[0][0]
        a2 =  rand_points[1][1] - rand_points[0][1]
        a3 =  rand_points[1][2] - rand_points[0][2]
        A = np.array([[a1,a2,a3]])

        #compute A = point2 - point0
        b1 =  rand_points[2][0] - rand_points[0][0]
        b2 =  rand_points[2][1] - rand_points[0][1]
        b3 =  rand_points[2][2] - rand_points[0][2]
        B = np.array([[b1,b2,b3]])

        # compute the plane coefficients c[0],c[1],c[2],D
        C = np.cross(A,B)
        D = -(C[0][0]*rand_points[1][0] + C[0][1]*rand_points[1][1] + C[0][2]*rand_points[1][2])
        return C[0][0],C[0][1],C[0][2],D

    #select the inliers based on the threshold distance(0.02)
    def inliers(self,a,b,c,d):
        a,b,c,d,dist = self.distance(a,b,c,d)
        plane_coeff = [a,b,c,d]
        index = np.where(np.abs(dist) <= 0.02)[0]
        inlier_points = self.pcd[index]
        return inlier_points, plane_coeff

    #calculate the distance from the plane
    def distance(self,a,b,c,d):
        num = a*self.pcd[:,0] + b*self.pcd[:,1] + c*self.pcd[:,2] + d
        denom = math.sqrt((a*a)+(b*b)+(c*c))
        dist = num/denom
        return a,b,c,d,dist

    # visualize the plane with inliers
    def plane_visualize(self):
        final_inliers = self.ransac()
       # Creating dataset
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(final_inliers)
        o3d.io.write_point_cloud("inlier.pcd", p)
        inlier = o3d.io.read_point_cloud("inlier.pcd")
        o3d.visualization.draw_geometries([inlier])

# the main function
def main():
    #read the point cloud data
    pcd = o3d.io.read_point_cloud("record_00348.pcd")
    #convert it to array
    pcd = (np.asarray(pcd.points))

    planefit = fit_plane(pcd)
    planefit.plane_visualize()

if __name__ == "__main__":
    main()