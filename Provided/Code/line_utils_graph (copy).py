import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR,'../utils'))
from skimage.measure import LineModelND, ransac
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.linear_model import LinearRegression
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler,scale
from sklearn.neighbors import KDTree
from scipy import optimize
from data_utils import *
import pickle
import json
import scipy.spatial as spatial
import multiprocessing as mp
import open3d as o3d
from MST_adj_matrix import Graph

verbose = False
def log_string(LOG_FOUT,out_str,verbose=verbose):
    if LOG_FOUT is not None:
        LOG_FOUT.write(out_str + '\n')
        LOG_FOUT.flush()
    if verbose:
        print(out_str)

class myline:
    def __init__(self, origin, direction, end_points, inlier, outlier,inlier_density=None,radius=None,length=None):
        self.origin = origin
        self.direction = direction
        self.end_points = end_points
        self.inlier = inlier # indices to xyz,np.array
        self.outlier = outlier # indices to xyz,np.array
        self.intersection = {'end':{0:[],1:[]}, 'mid':[]}
        self.inlier_density = inlier_density #points/length（cm)
        self.radius = radius
        self.length = length #in meter
        self.inlier_pred = []


    def get_length(self):
        if self.length != None:
            return self.length
        else:
            self.length = np.linalg.norm((self.end_points[0]-self.end_points[1]))
            return self.length

    def update_length(self):
        self.length = np.linalg.norm((self.end_points[0] - self.end_points[1]))

    def add_intersection(self, joint_id,rel_location,endpoint_id=0):
        if rel_location == 'end':
            assert joint_id not in self.intersection['mid'], "joint_id %d already in mid intersection, cannot add to end!"%joint_id
            if joint_id not in self.intersection[rel_location][endpoint_id]:
                if isinstance(joint_id,(int, np.integer)):
                    self.intersection[rel_location][endpoint_id].append(joint_id)
                elif isinstance(joint_id,list):
                    self.intersection[rel_location][endpoint_id].extend(joint_id)
                else:raise ValueError('Unknown joint id type!',type(joint_id))
        elif rel_location == 'mid':
            assert joint_id not in self.intersection['end'][0], "joint_id %d already in end[0] intersection, cannot add to mid!"%joint_id
            assert joint_id not in self.intersection['end'][1], "joint_id %d already in end[1] intersection, cannot add to mid!"%joint_id
            if joint_id not in self.intersection[rel_location]:
                if isinstance(joint_id, (int, np.integer)):
                    self.intersection[rel_location].append(joint_id)
                elif isinstance(joint_id, list):
                    self.intersection[rel_location].extend(joint_id)
        else:
            raise ValueError("Unknown intersection reletive loctaion!")

    def remove_intersection(self,joint_id,rel_location=None,endpoint_id=0):
        if rel_location is not None:
            if rel_location == 'end':
                self.intersection[rel_location][endpoint_id].remove(joint_id)
            elif rel_location == 'mid':
                self.intersection[rel_location].remove(joint_id)
        else:
            if joint_id in self.intersection['end'][0]:
                self.intersection['end'][0].remove(joint_id)
            elif joint_id in self.intersection['end'][1]:
                self.intersection['end'][1].remove(joint_id)
            elif joint_id in self.intersection['mid']:
                self.intersection['mid'].remove(joint_id)
            else:
                raise ValueError(f'joint {joint_id} not in intersection, cannot remove!')


    def get_inlier_density(self):
        if self.inlier_density!=None:
            return self.inlier_density
        else:
            self.inlier_density = self.inlier.shape[0]/self.get_length()/100
            return self.inlier_density
    def update_inlier_density(self):
        self.inlier_density = self.inlier.shape[0]/self.get_length()/100

    def assign_radius(self,radius):
        '''unit:meter'''
        self.radius = radius


class myjoint:
    def __init__(self, position,type=None,connected_line = None):
        '''type = "end/T/cross/triple, cross:all end jointed, triple:one main branch(mid-jointed), >=2 sub-branch (end-jointed)"
        connected_line: [line_id,id...]'''
        self.position = position
        self.type = type
        self.connected_lines = connected_line if connected_line is not None else []

    def add_line(self,line_id):
        '''line_id: [id,id...]'''
        if isinstance(line_id, (int, np.integer)):
            self.connected_lines.append(line_id)
        elif isinstance(line_id, list):
            self.connected_lines.extend(line_id)
    def assign_type(self, type):
        self.type = type
    def get_type(self):
        return self.type

class mypipe():
    def __init__(self,line_id=None, joint_id=None):
        '''line_id/joint_id = [id,...]'''
        self.lines = line_id if line_id is not None else []
        self.joints = joint_id if joint_id is not None else []

    def add_line(self,line_id):
        '''line_id = [id]'''
        if isinstance(line_id, (int, np.integer)):
            self.lines.append(line_id)
        elif isinstance(line_id, list):
            self.lines.extend(line_id)

    def add_joint(self,joint_id):
        '''joint_id = [id]'''
        if isinstance(joint_id, (int, np.integer)):
            self.joints.append(joint_id)
        elif isinstance(joint_id, list):
            self.joints.extend(joint_id)


def line_ransac(xyz,ransac_residual_ths, outlier_min_residual, min_line_length = None, min_inlier_density = None,transfer_indice=False,xyz_idx=None, assign_radius=False, radius=None, min_samples=2, max_trials=1000,get_full_length=False,return_outlier=False):
    '''if outlier_min_residual is None, keep all points; else discard points outside outlier_min_residual
    line_length: cm  inlier_density: #points/cm
    transfer_indice: True when xyz is not xyz_orig, input xyz_idx of xyz indice to xyz_orig
    get_full_length: True for merging lines, in case of not perfect straight pipe, assume all xyz are inliers and extend ransac line to get full length'''

    model_robust, inliers = ransac(xyz, LineModelND, min_samples=min_samples,
                                   residual_threshold=ransac_residual_ths, max_trials=max_trials)
    outliers = inliers == False
    outlier = outliers.nonzero()  # indice to xyz
    inlier = inliers.nonzero()[0]
    num_inlier = inlier.shape[0]
    if outlier_min_residual is not None:
        neighbor = ((model_robust.residuals(xyz) <= outlier_min_residual) & (
            model_robust.residuals(xyz) > ransac_residual_ths)).nonzero()[0]
    else:
        neighbor = outlier[0]
    if assign_radius:
        assert radius is not None,"Radius is None, can't assign radius!"
        if outlier_min_residual is None:
            r = np.average(radius)
        else:
            r = np.average(radius[inlier])
    if transfer_indice:
        assert xyz_idx is not None,"Provide xyz_idx in order to transfer indice!"
        inlier = xyz_idx[inlier] #indice to xyz_orig
        neighbor = xyz_idx[neighbor] #indice to xyz_orig
    if get_full_length:
        inliers = np.ones(xyz.shape[0],dtype=bool)

    direction = model_robust.params[1]
    principal_direction = np.argmax(np.abs(direction))
    origin = model_robust.params[0]
    zz = xyz[inliers][:, principal_direction]
    inlier_min_point = xyz[inliers][np.argmin(zz)]
    inlier_max_point = xyz[inliers][np.argmax(zz)]
    line_min_point = origin + direction * (np.dot((inlier_min_point - origin), direction))
    line_max_point = origin + direction * (np.dot((inlier_max_point - origin), direction))
    length = np.linalg.norm(line_max_point - line_min_point) # 1cm spacing
    if length > 0:
        inlier_density = num_inlier/length/100
    else:
        inlier_density = 0
    # print('line_ransac  length %.4f  inlier density %.5f'%(length,inlier_density))

    if min_inlier_density is not None:
        if inlier_density < min_inlier_density:
            if return_outlier:
                return None,neighbor
            else: return None
    if min_line_length is not None :
        if length > min_line_length:
            line = myline(model_robust.params[0], model_robust.params[1], [line_min_point, line_max_point], inlier, neighbor,inlier_density=inlier_density)
            if assign_radius:
                line.assign_radius(r)
            if return_outlier: return line,neighbor
            else:return line
    else: # do not consider length
        line = myline(model_robust.params[0], model_robust.params[1], [line_min_point, line_max_point], inlier,neighbor,inlier_density=inlier_density,length=length)
        if assign_radius:
            line.assign_radius(r)
        if return_outlier: return line,neighbor
        else:return line

def line_angle(dir1,dir2,return_degree=False):
    cross = np.dot(dir1, dir2)
    angle = np.arccos(np.clip(np.abs(cross),-1.0,1.0))
    if return_degree:
        return np.rad2deg(angle)
    else:return angle

def point2line_distance(l1p1, l1p2, l2p1):
    '''line1: pass a,b; line2: pass c. Same as point2line_vertical_distance
    return distance of point c to line1'''
    distance = np.linalg.norm(np.cross((l2p1 - l1p1), (l2p1 - l1p2))) / np.linalg.norm((l1p1 - l1p2))
    return distance

def points2line_distance(a,b,pts):
    '''line: pass a,b
    return distances of pts to line'''
    a = np.reshape(a,(1,3))
    b = np.reshape(b,(1,3))
    pts = np.reshape(pts,(-1,3))
    distances = np.linalg.norm(np.cross((pts - a), (pts - b)),axis=-1) / np.linalg.norm((a - b))
    return distances


def line2line_distance(dir1,dir2,l1p,l2p):
    '''shortest distance between skew lines '''
    cross = (np.cross(dir1, dir2)) / (np.linalg.norm(dir1) * np.linalg.norm(dir2))
    distance = np.abs(np.dot(cross,(l1p-l2p)))
    return distance

def point2point_distance(p1,p2):
    return np.linalg.norm((p1-p2))

def points2point_distance(p1,p2):
    p1 = np.reshape(p1,(-1,3))
    p2 = np.reshape(p2,(-1,3))
    return np.linalg.norm((p1-p2),axis=-1)


def nearest_points(dir1, dir2, l1p, l2p):
    '''dir1&2: direction unit vector of line1&2
    return point c1 on line1 that is nearest to line 2 and point c2 on line2 that is nearest to line 1
    points may be on extention of line segment'''
    # cross = (np.cross(dir1,dir2))/(np.linalg.norm(dir1)*np.linalg.norm(dir2))
    # c1 = l1p + np.dot((l2p - l1p), dir1) * dir1
    # c2 = l2p + np.dot((l1p - l2p), dir2) * dir2

    A = np.array([[np.dot(dir1,dir1),-np.dot(dir1,dir2)],[np.dot(dir1,dir2),-np.dot(dir2,dir2)]])
    B = np.array([[np.dot((l2p-l1p),dir1)],[np.dot((l2p-l1p),dir2)]])
    X = np.linalg.lstsq(A,B,rcond=None)[0]
    return l1p+dir1*X[0],l2p+dir2*X[1]

def endpt_pair(cur_line_endpt,check_line_endpt):
    '''a&b, from l1, c&d from l2, return pair index of a&b to c&d, e.g. [0,1] = a-c,b-d'''
    a_to_l2 = [point2point_distance(cur_line_endpt[0],check_line_endpt[0]),point2point_distance(cur_line_endpt[0],check_line_endpt[1])]
    b_to_l2 = [point2point_distance(cur_line_endpt[1],check_line_endpt[0]),point2point_distance(cur_line_endpt[1],check_line_endpt[1])]
    z = np.array([a_to_l2,b_to_l2])
    zmin = np.argmin(z,axis=-1)
    if zmin[0] == zmin[1]:
        if z[0,zmin[0]] < z[1,zmin[1]]: #keep closer point's pair
            zmin[1] = 1-zmin[1]
        else:
            zmin[0] = 1-zmin[0]
    return zmin





def point2line_projection(dir1,l1p1,l2p):
    '''projection of l2p on l1, scalar'''
    if len(l2p.shape) > 1: #(N,3)
        dir1 = np.reshape(dir1,(-1,3))
        l1p1 = np.reshape(l1p1,(-1,3))
        diff = l2p - l1p1
        projection = np.dot(dir1,(l2p - l1p1))
    else:
        projection = np.dot(dir1,(l2p - l1p1))
    return projection


def points2line_projection(dir1,l1p1,l2p):
    '''projection of l2p on l1, scalar, anchor at l1p1'''
    if len(l2p.shape) > 1: #(N,3)
        dir1 = np.reshape(dir1,(-1,3))
        l1p1 = np.reshape(l1p1,(-1,3))
        l2p = np.reshape(l2p,(-1,3))
        projection = np.dot((l2p - l1p1),np.transpose(dir1))
    else:
        projection = np.dot(dir1,(l2p - l1p1))
    return projection

def point2line_vertical_distance(dir1,l1p1,l2p):
    '''vertical distance of l2p to l1. vertical relative to dir 1'''
    return np.linalg.norm(np.cross(dir1,(l2p - l1p1)))

def point2lines_vertical_distance(dir1,l1p1,l2p):
    '''vertical distance of l2p to l1. vertical relative to dir 1'''
    l1p1 = np.reshape(l1p1,(-1,3))
    dir1 = np.reshape(dir1,(-1,3))
    l2p = np.reshape(l2p,(-1,3))
    return np.linalg.norm(np.cross(dir1,(l2p - l1p1)),axis=-1)

def check_overlap(dir1, l1p1, l1p2,projection,):
    t = projection/np.dot((l1p2-l1p1),dir1)
    if (t>=0) and (t<=1):
        return True
    else:
        return False

def pointline_overlap(l1p1, l1p2, l2p):
    '''check if l2p falls within l1's segment p1p2, return bool'''
    t = np.dot((l2p-l1p1),(l1p2-l1p1))/np.dot((l1p2 - l1p1),(l1p2 - l1p1))
    if (t>=0) and (t<=1):
        return True
    else:
        return False


def pointsline_overlap(l1p1, l1p2, l2p, return_t=False):
    '''check if l2p falls within l1's segment p1p2, return bool'''
    l1p1 = np.reshape(l1p1,(-1,3))
    l1p2 = np.reshape(l1p2,(-1,3))
    t = np.dot((l2p-l1p1),np.transpose(l1p2-l1p1))/np.dot((l1p2 - l1p1),np.transpose(l1p2 - l1p1))
    t = np.reshape(t,-1)
    if return_t:
        return (t>=0) & (t<=1), t
    else:
        return (t>=0) & (t<=1)



def line_overlap(l1,l2):
    a = l1.end_points[0]
    b = l1.end_points[1]
    c = l2.end_points[0]
    d = l2.end_points[1]
    overlap = np.array((pointline_overlap(a, b, c), pointline_overlap(a, b, d), pointline_overlap(c,d,a),pointline_overlap(c, d, b)))
    return overlap.any()

def line_crossing(l1:myline,l2:myline,intersect1=None,intersect2=None):
    '''intersect1 on l1, intersect2 on l2'''
    if intersect1 is None or intersect2 is None:
        intersect1,intersect2=nearest_points(l1.direction, l2.direction, l1.end_points[0], l2.end_points[0])
    if pointline_overlap(l1.end_points[0],l1.end_points[1],intersect1) and pointline_overlap(l2.end_points[0],l2.end_points[1],intersect2):
        return True
    else:
        return False

def verify_intersection(line, point, p2p_dist_ths = 0.03,p2l_dist_ths = 0.03):
    '''true intersection if distance of point to end points is smaller than threshold (end intersect), or falls within end points (mid intersect)'''
    dist = min(point2point_distance(point,line.end_points[0]),point2point_distance(point,line.end_points[1]))
    if dist < p2p_dist_ths:
        return "end"
    elif (pointline_overlap(line.end_points[0], line.end_points[1], point)) & (point2line_distance(line.end_points[0], line.end_points[1], point) < p2l_dist_ths):
        return "mid"
    else:
        return False

def get_nearest_endpoints_pair_non_overlap(line1, line2):
    '''For non overlappint lines, check endpoints distance
    return nearest pair of end points'''
    a = line1.end_points[0]
    b = line1.end_points[1]
    c = line2.end_points[0]
    d = line2.end_points[1]
    output = []
    if point2point_distance(a, c) < point2point_distance(b,c):
        output.extend([0])
    else:
        output.extend([1])
    if point2point_distance(c, a) < point2point_distance(d,a):
        output.extend([0])
    else:
        output.extend([1])
    return output

def get_nearest_endpoints_pair_overlap(line1, line2):
    '''For overlapping lines, check endpoint to line vertical distance
    return nearest pair of end points'''
    a = line1.end_points[0]
    b = line1.end_points[1]
    c = line2.end_points[0]
    d = line2.end_points[1]
    output = []
    if point2line_vertical_distance(line2.direction,c,a) < point2line_vertical_distance(line2.direction,c,b):
        output.extend([0])
    else:
        output.extend([1])
    if point2line_vertical_distance(line1.direction, a, c) < point2line_vertical_distance(line1.direction, a, d):
        output.extend([0])
    else:
        output.extend([1])
    return output

    # dist = np.array((point2point_distance(a,c),point2point_distance(b,c),point2point_distance(a,d),point2point_distance(b,d)))
    # min_dist = np.argmin(dist)
    # if min_dist == 0:
    #     return [0, 0]
    # elif min_dist == 1:
    #     return [1, 0]
    # elif min_dist == 2:
    #     return [0, 1]
    # elif min_dist == 3:
    #     return [1, 1]

def project2plane(plane_p,plane_n,p):
    '''query point p projection on plane with normal plane_n pass through point plane_p'''
    d_mag = np.dot((p-plane_p),np.transpose(plane_n))
    d = d_mag*plane_n
    return p-d

def break_lines(lines,xyz_orig,radius_orig,DBSCAN_eps = 0.03, DBSCAN_min_samples=10, min_cluster_points = 30,ransac_residual_ths=0.005,outlier_min_residual=None,min_line_length=6,min_inlier_density=None,use_outlier=True):
    '''break lines with more than 1 cluster of inliers; discard segments with less than 30 inliers'''
    num_line = len(lines)
    break_list = []
    prune_list = []
    inlier_cluster = []
    for cur_Lid in range(num_line):
        if not use_outlier:
            X = xyz_orig[lines[cur_Lid].inlier]
            X_indice = lines[cur_Lid].inlier
        else:
            X = np.concatenate((xyz_orig[lines[cur_Lid].inlier],xyz_orig[lines[cur_Lid].outlier]),axis=0)
            X_indice = np.concatenate((lines[cur_Lid].inlier , lines[cur_Lid].outlier),axis=0)
        # check if inliers have separate clusters
        dbscan = DBSCAN(eps=DBSCAN_eps, min_samples=DBSCAN_min_samples) #clusters less than 30 points are discarded
        clusters = dbscan.fit_predict(X)
        num_clusters = max(clusters) + 1 #excluding noises where clusters=-1
        if max(clusters)==-1: # only have noise
            prune_list.append(cur_Lid)
            print('prune line %d'%cur_Lid)
        elif max(clusters) >0: # have at least 2 clusters
            break_list.append(cur_Lid)
            print('break line%d'%cur_Lid,' %d clusters'%num_clusters)
            for i in range(0,num_clusters):
                cluster_inx = (clusters == i).nonzero() #indice to X
                # cluster_inx = lines[cur_Lid].inlier[cluster_inx] #indice to xyz_orig
                cluster_inx = X_indice[cluster_inx] #indice to xyz_orig

                if cluster_inx.shape[0] > min_cluster_points:
                    inlier_cluster.append(cluster_inx)
                else:
                    print('   line %d cluster %d has %d points -- pruned'%(cur_Lid,i,cluster_inx.shape[0]))
        elif max(clusters)==0: # have only 1 cluster
            if min(clusters) == -1: # have noise
                break_list.append(cur_Lid)
                cluster_inx = (clusters == 0).nonzero()  # indice to X
                # cluster_inx = lines[cur_Lid].inlier[cluster_inx] #indice to xyz_orig
                cluster_inx = X_indice[cluster_inx]  # indice to xyz_orig
                if cluster_inx.shape[0] < min_cluster_points:
                    print('line %d has 1 cluster but too little points, pruned' % cur_Lid)
                else:
                    inlier_cluster.append(cluster_inx)
                    print('line %d has 1 cluster, noises removed' % cur_Lid)
            else: # do not have noise. all clusters == 0
                print('line %d not breaked' % cur_Lid)
    lines = [l for i, l in enumerate(lines) if i not in break_list and i not in prune_list] #remove breaked lines

    for cluster in inlier_cluster:
        newLine = line_ransac(xyz_orig[cluster],ransac_residual_ths,outlier_min_residual,min_line_length=min_line_length,min_inlier_density=min_inlier_density)
        if newLine is not None:
            newLine.inlier = cluster[newLine.inlier]
            newLine.outlier = cluster[newLine.outlier]
            radius = np.average(radius_orig[newLine.inlier])
            newLine.assign_radius(radius)
            lines.append(newLine)
            ##TODO: check inlier_density respective to radius?
            # print('inlier density/radius (#/R #/cm2) ',newLine.inlier_density/(newLine.radius*100))
            # if newLine.inlier_density > radius * 100 * 3.1:
            #     # newLine = None
            #     pass

    return lines

def break_lines_ind(line):
    inlier_cluster = []
    X = line.inlier
    # check if inliers have separate clusters
    dbscan = DBSCAN(eps=0.05, min_samples=30) #clusters less than 30 points are discarded
    clusters = dbscan.fit_predict(X)
    num_clusters = max(clusters) - min(clusters) + 1
    if num_clusters > 1:
        for i in range(num_clusters):
            cluster = X[clusters == min(clusters) + i]
            if cluster.shape[0] > 30:
                inlier_cluster.append(cluster)
                # save_ply(cluster,os.path.join(sim_data_path,'results/breaked_line%d_cluster%d.ply'%(cur_Lid,i)))
    breaked_lines = []
    for cluster in inlier_cluster:
        breaked_lines.append(line_ransac(cluster,0.005,None,min_line_length=3))
    breaked_lines = [l for l in breaked_lines if l is not None] #remove short lines
    for line in breaked_lines:
        print(line.get_length())
    return breaked_lines

def check_linearity(tree,query_point,radius,xyz_orig):
    '''return: ratio: 1 being perfect linear; '''
    ind = tree.query_radius(query_point,radius,count_only=False)
    ratio = 0
    z = 0
    balance_ratio = 0
    for i in ind:
        pca = decomposition.PCA(n_components=3)
        pca_fitted = pca.fit(xyz_orig[i])
        eigenvalues_ratio = pca_fitted.explained_variance_ratio_
        ratio += eigenvalues_ratio[0]
        # remaining = (1-eigenvalues_ratio[0])/2 # lum
        # balance_ratio += abs(eigenvalues_ratio[1]-remaining)/ remaining # smaller, better
        # print(eigenvalues_ratio)
        # print(abs(eigenvalues_ratio[1]-remaining)/ remaining)
        # save_ply(xyz_orig[i],os.path.join(BASE_DIR,'temp%d.ply'%z))
        z+=1
    ratio = ratio/len(ind)
    # balance_ratio /= len(ind)
    # print('balance_ratio ',balance_ratio)
    # print('ratio ',ratio)
    return ratio

def check_pred_data_linearity(pred, lines, pred_tree, linearity_ths=0.9):
    num_line = len(lines)
    prune_list = []
    principal_eigen_arr = []
    planarity_arr = []
    linear_ratio_arr = []
    scatter_arr = []
    for cur_Lid in range(num_line):
        cur_line = lines[cur_Lid]
        radius = cur_line.radius
        inlier = cur_line.inlier
        num_inlier = inlier.shape[0]
        # sample_pt = np.random.choice(inlier, max(int(num_inlier * 0.01), min(100, num_inlier)), replace=False)
        sample_pt = inlier
        # mid_pt = (cur_line.end_points[0] + cur_line.end_points[1])/2
        # quad_pt1 = cur_line.end_points[0] + (cur_line.end_points[1]-cur_line.end_points[0])*0.25
        # quad_pt2 = cur_line.end_points[0] + (cur_line.end_points[1]-cur_line.end_points[0])*0.75
        # third_pt1 = cur_line.end_points[0] + (cur_line.end_points[1]-cur_line.end_points[0])*0.33
        # third_pt2 = cur_line.end_points[0] + (cur_line.end_points[1]-cur_line.end_points[0])*0.67
        # scan_ind = pred_tree.query(np.stack((mid_pt, quad_pt1, quad_pt2, third_pt1, third_pt2)), k=1) #nearest scan point
        ind = pred_tree.query_radius(pred[sample_pt], radius*2, count_only=False)

        # num_inlier = cur_line.inlier.shape[0]
        # sample_ind = np.random.choice(num_inlier,max(int(num_inlier*0.001),10))
        # sample_ind = cur_line.inlier[sample_ind] #indice to prediction
        # inlier_ind = pred2scan_indice[sample_ind] #indice to scan data
        # # np.savetxt(os.path.join(BASE_DIR,'temp_center.txt'),scan[inlier_ind])
        # # np.savetxt(os.path.join(BASE_DIR,'temp_inlier.txt'),xyz_orig[sample_ind])
        # ind = scan_tree.query_radius(scan[inlier_ind], radius, count_only=False)

        # ## make sure all points have at least 3 neighbors
        # for i,neighbor_id in enumerate(ind):
        #     if neighbor_id.shape[0] <3:
        #         resample_ind = sample_ind[i]
        #         while resample_ind in sample_ind:
        #             resample_ind = np.random.choice(num_inlier)
        #         ind[i] = scan_tree.query_radius(np.reshape(scan[pred2scan_indice[resample_ind]],(-1,3)), radius, count_only=False)
        # if ind.shape[0] <3:
        #     resample_ind = sample_ind[i]
        #     while resample_ind in sample_ind:
        #         resample_ind = np.random.choice(num_inlier)
        #     ind[i] = scan_tree.query_radius(np.reshape(scan[pred2scan_indice[resample_ind]],(-1,3)), radius, count_only=False)
        ratio1 = 0
        ratio2 = 0
        ratio3 = 0
        linear_ratio = 0
        planarity = 0
        scattering = 0
        points = []
        for i in ind:
            points.append(pred[i])
        points = np.concatenate(points,axis=0)

        pca = decomposition.PCA(n_components=3)
        pca_fitted = pca.fit(points)
        # np.savetxt(os.path.join(BASE_DIR,'temp_%d.txt'%z),scan[i])
        eigenvectors = pca_fitted.components_
        points_trans = np.dot(points,np.transpose(eigenvectors))
        ## scale translated points along principal direction, 0-centered, unit variance and legnth = 2
        points_trans_x = scale(points_trans[:,0],copy=True)
        points_trans[:,0] = points_trans_x/(max(points_trans_x)-min(points_trans_x))
        # save_ply(points_trans,os.path.join(BASE_DIR,'temp_points_trans.ply'))

        pca = decomposition.PCA(n_components=3)
        pca_fitted = pca.fit(points_trans)
        eigenvalues_ratio = pca_fitted.explained_variance_ratio_
        ratio1 += eigenvalues_ratio[0]
        principal_eigen_arr.append(eigenvalues_ratio[0])
        ratio2 += eigenvalues_ratio[1]
        ratio3 += eigenvalues_ratio[2]
        planarity += (eigenvalues_ratio[1]-eigenvalues_ratio[2])/eigenvalues_ratio[0]
        planarity_arr.append(planarity)
        scattering += eigenvalues_ratio[2]/eigenvalues_ratio[0]
        scatter_arr.append(scattering)
        linear_ratio += (eigenvalues_ratio[0]-eigenvalues_ratio[1])/eigenvalues_ratio[0]
        linear_ratio_arr.append(linear_ratio)
        # print(eigenvalues_ratio)
        # remaining = (1-eigenvalues_ratio[0])/2 # lum
        # balance_ratio += abs(eigenvalues_ratio[1]-remaining)/ remaining # smaller, better
        # print(eigenvalues_ratio)
        # print(abs(eigenvalues_ratio[1]-remaining)/ remaining)
        # save_ply(xyz_orig[i],os.path.join(BASE_DIR,'temp%d.ply'%z))
        # ratio1 /= len(ind)
        # ratio2 /= len(ind)
        # ratio3 /= len(ind)
        # ratio21 /= len(ind)
        # ratio32 /= len(ind)
        # linear_ratio /= len(ind)
        if linear_ratio < linearity_ths:
            prune_list.append(cur_Lid)
            print(' %d pruned ratio1 %.5f ratio2 %.5f ratio3 %.5f linear_ratio %.5f planarity %.5f scattering %.5f'%(cur_Lid,ratio1,ratio2,ratio3,linear_ratio,planarity,scattering))
        else:
            print(' %d remained ratio1 %.5f ratio2 %.5f ratio3 %.5f linear_ratio %.5f planarity %.5f scattering %.5f' % (cur_Lid, ratio1, ratio2, ratio3, linear_ratio, planarity, scattering))
    principal_eigen_arr=np.array(principal_eigen_arr)
    percentile = np.percentile(principal_eigen_arr,range(0,100,5))
    planarity_arr = np.array(planarity_arr)
    percentile_21 = np.percentile(planarity_arr,range(0,100,5))
    linear_ratio_arr = np.array(linear_ratio_arr)
    percentile_li = np.percentile(linear_ratio_arr, range(0, 100, 5))
    scatter_arr = np.array(scatter_arr)
    percentile_sc = np.percentile(scatter_arr, range(0, 100, 5))
    np.set_printoptions(suppress=True)
    print('principal eigenvector ratio ','mean %.5f'%(np.mean(principal_eigen_arr)),' percentile ',percentile)
    np.set_printoptions(suppress=True)
    print('eigenvector2/1 ratio ','mean %.5f'%(np.mean(planarity_arr)),' percentile ',percentile_21)

    print('principal ',np.where(principal_eigen_arr<percentile[2]))
    print('planarity ',np.where(planarity_arr>percentile_21[-3]))
    print('planarity ',np.where(planarity_arr>percentile_21[-2]))
    print('linear ',np.where(linear_ratio_arr<percentile_li[2]))
    print('scatter ',np.where(scatter_arr>percentile_sc[-3]))

    lines = [l for i, l in enumerate(lines) if i not in prune_list]
    return lines
#72,126,98,176
# 176 remained ratio1 0.98146 ratio2 0.01297 ratio3 0.00558 linear_ratio 0.98679 planarity 0.00753 scattering 0.00568
# 126 remained ratio1 0.98265 ratio2 0.01485 ratio3 0.00250 linear_ratio 0.98489 planarity 0.01257 scattering 0.00254
# 98 remained ratio1 0.99210 ratio2 0.00512 ratio3 0.00278 linear_ratio 0.99484 planarity 0.00235 scattering 0.00280
# 72 remained ratio1 0.98756 ratio2 0.00727 ratio3 0.00517 linear_ratio 0.99264 planarity 0.00213 scattering 0.00523

def check_scan_data_linearity(scan,pred2scan_indice, lines, scan_tree, planarity_ths=0.9):
    num_line = len(lines)
    prune_list = []
    planarity_arr = []
    scatter_arr=[]
    linear_ratio_arr=[]
    for cur_Lid in range(num_line):
        cur_line = lines[cur_Lid]
        radius = cur_line.radius * 2.5
        inlier = cur_line.inlier
        num_inlier = inlier.shape[0]
        # sample_pt = np.random.choice(inlier, max(int(num_inlier * 0.01), min(100, num_inlier)), replace=False)
        sample_pt = inlier
        scan_pt = scan[pred2scan_indice[sample_pt]]
        # save_ply(scan_pt,os.path.join(BASE_DIR,'temp_scan_pt%d.ply'%cur_Lid))

        # mid_pt = (cur_line.end_points[0] + cur_line.end_points[1])/2
        # quad_pt1 = cur_line.end_points[0] + (cur_line.end_points[1]-cur_line.end_points[0])*0.25
        # quad_pt2 = cur_line.end_points[0] + (cur_line.end_points[1]-cur_line.end_points[0])*0.75
        # third_pt1 = cur_line.end_points[0] + (cur_line.end_points[1]-cur_line.end_points[0])*0.33
        # third_pt2 = cur_line.end_points[0] + (cur_line.end_points[1]-cur_line.end_points[0])*0.67
        # scan_ind = scan_tree.query(np.stack((mid_pt, quad_pt1, quad_pt2, third_pt1, third_pt2)), k=1) #nearest scan point
        # ind = scan_tree.query_radius(np.reshape(scan[scan_ind[1]], (-1, 3)), radius, count_only=False)

        # num_inlier = cur_line.inlier.shape[0]
        # sample_ind = np.random.choice(num_inlier,max(int(num_inlier*0.001),10))
        # sample_ind = cur_line.inlier[sample_ind] #indice to prediction
        # inlier_ind = pred2scan_indice[sample_ind] #indice to scan data
        # # np.savetxt(os.path.join(BASE_DIR,'temp_center.txt'),scan[inlier_ind])
        # # np.savetxt(os.path.join(BASE_DIR,'temp_inlier.txt'),xyz_orig[sample_ind])
        # ind = scan_tree.query_radius(scan[inlier_ind], radius, count_only=False)

        # ## make sure all points have at least 3 neighbors
        # for i,neighbor_id in enumerate(ind):
        #     if neighbor_id.shape[0] <3:
        #         resample_ind = sample_ind[i]
        #         while resample_ind in sample_ind:
        #             resample_ind = np.random.choice(num_inlier)
        #         ind[i] = scan_tree.query_radius(np.reshape(scan[pred2scan_indice[resample_ind]],(-1,3)), radius, count_only=False)
        # if ind.shape[0] <3:
        #     resample_ind = sample_ind[i]
        #     while resample_ind in sample_ind:
        #         resample_ind = np.random.choice(num_inlier)
        #     ind[i] = scan_tree.query_radius(np.reshape(scan[pred2scan_indice[resample_ind]],(-1,3)), radius, count_only=False)
        ratio1 = 0
        ratio2 = 0
        ratio3 = 0
        planar_ratio = 0

        # for i in ind:
        # pca = decomposition.PCA(n_components=3)
        # pca_fitted = pca.fit(scan_pt)
        # # np.savetxt(os.path.join(BASE_DIR,'temp_%d.txt'%z),scan[i])
        # # z+=1
        # eigenvalues_ratio = pca_fitted.explained_variance_ratio_
        pca = decomposition.PCA(n_components=3)
        pca_fitted = pca.fit(scan_pt)
        # np.savetxt(os.path.join(BASE_DIR,'temp_%d.txt'%z),scan[i])
        eigenvectors = pca_fitted.components_
        points_trans = np.dot(scan_pt, np.transpose(eigenvectors))
        ## scale translated points along principal direction, 0-centered, unit variance and legnth = 2
        points_trans_x = scale(points_trans[:, 0], copy=True)
        points_trans[:, 0] = points_trans_x / (max(points_trans_x) - min(points_trans_x))
        # save_ply(points_trans,os.path.join(BASE_DIR,'temp_points_trans.ply'))
        pca = decomposition.PCA(n_components=3)
        pca_fitted = pca.fit(points_trans)
        eigenvalues_ratio = pca_fitted.explained_variance_ratio_
        ratio1 += eigenvalues_ratio[0]
        ratio2 += eigenvalues_ratio[1]
        ratio3 += eigenvalues_ratio[2]
        planarity = (eigenvalues_ratio[1] - eigenvalues_ratio[2]) / eigenvalues_ratio[0]
        planarity_arr.append(planarity)
        scattering = eigenvalues_ratio[2] / eigenvalues_ratio[0]
        scatter_arr.append(scattering)
        linear_ratio = (eigenvalues_ratio[0] - eigenvalues_ratio[1]) / eigenvalues_ratio[0]
        linear_ratio_arr.append(linear_ratio)

        if planar_ratio > planarity_ths:
            prune_list.append(cur_Lid)
            print(' %d pruned ratio1 %.5f ratio2 %.5f ratio3 %.5f linear_ratio %.5f planarity %.5f scattering %.5f'%(cur_Lid,ratio1,ratio2,ratio3,linear_ratio,planarity,scattering))
        else:
            print(' %d remained ratio1 %.5f ratio2 %.5f ratio3 %.5f linear_ratio %.5f planarity %.5f scattering %.5f' % (cur_Lid, ratio1, ratio2, ratio3, linear_ratio, planarity, scattering))
    lines = [l for i, l in enumerate(lines) if i not in prune_list]
    print('linear<0.99 ',np.nonzero(np.less(linear_ratio,0.99)))
    return lines

def merge_lines_ept_search(lines, xyz_orig, radius_orig, paraline_distance_ths=0.02, contline_distance_ths=0.1, angle_ths =20 / 180 * np.pi, ransac_residual_ths = 0.005, outlier_min_residual=0.015,min_inlier_density=None, line_fit_method='ransac', radius_difference = 0.2, ths_in_radius_multiple = False,log_file=None,max_itr = 5):
    '''only check lines with endpoints within some distance
    iterate until no lines are merged'''
    LOG_FOUT = open(log_file,'w+')
    print('Start merge lines')
    has_merge = True
    itr = 1
    while has_merge and itr <=max_itr: # iterate until no lines are merged
        has_merge = False
        log_string(LOG_FOUT,f'\n\n{itr} time merge!!\n')
        itr+=1

        num_line = len(lines)
        endpoint_list = [l.end_points for l in lines]
        endpoint_list = np.vstack(endpoint_list) #(num_lines*2,3)
        radius_list = np.vstack([l.radius for l in lines]) #(num_lines,)

        dir_list = np.vstack([l.direction for l in lines])
        endpoint1_list = endpoint_list[[i for i in range(len(endpoint_list)) if i%2]]
        pool = mp.Pool((mp.cpu_count()))
        mid_target = pool.starmap(point2lines_vertical_distance,[(dir_list,endpoint1_list,endpoint_list[i]) for i in range(len(endpoint_list))])
        pool.close()
        # regions = [np.concatenate((np.where(mid_target[i]<radius_list[i//2]*max(paraline_distance_ths,contline_distance_ths)*2)[0],np.where(mid_target[i+1]<radius_list[i//2]*max(paraline_distance_ths,contline_distance_ths)*2)[0])) for i in range(0,len(mid_target),2)] #list of arrays(N,) of line id, len=num_lines content:line id
        regions = [np.concatenate((np.where(mid_target[i]<radius_list[i//2]*paraline_distance_ths*2)[0],np.where(mid_target[i+1]<radius_list[i//2]*paraline_distance_ths*2)[0])) for i in range(0,len(mid_target),2)] #list of arrays(N,) of line id, len=num_lines content:line id


        check_map = np.zeros((num_line,num_line),dtype=int)
        mergelist = np.zeros((num_line, num_line), dtype=bool)
        for i,region in tqdm(enumerate(regions), desc='check lines merging'):
            cur_line_merge_list = []
        # for i,region in enumerate(regions):
            cur_Lid = i

            cur_line = lines[cur_Lid]
            a = cur_line.end_points[0]
            b = cur_line.end_points[1]
            region = np.unique(region)
            region = region[region != cur_Lid]
            #check angle in batch
            check_angle=np.array([lines[i].direction for i in region])
            cross = np.dot(cur_line.direction, np.transpose(np.reshape(check_angle,(-1,3))))
            angle = np.arccos(np.clip(np.abs(cross),-1.0,1.0))
            region = region[angle<angle_ths]
            angle=angle[angle<angle_ths]
            for idx,j in enumerate(region):
                check_id = j.item() #convert to python int type, for json output use
                if check_map[cur_Lid,check_id]:
                    # log_string(LOG_FOUT," line %d and %d is already checked, skip"%(cur_Lid,check_id))
                    continue
                else:
                    check_map[cur_Lid,check_id] = 1
                    check_map[check_id,cur_Lid] = 1
                check_line = lines[check_id]
                longer_line = cur_line if cur_line.get_length()>check_line.get_length() else check_line
                merged = False
                c = check_line.end_points[0]
                d = check_line.end_points[1]
                # overlap = (pointline_overlap(a, b, c)) | (pointline_overlap(a, b, d))| (pointline_overlap(c,d,a))| (pointline_overlap(c, d, b))
                overlap = np.array((pointline_overlap(a, b, c), pointline_overlap(a, b, d), pointline_overlap(c,d,a),pointline_overlap(c, d, b)))
                has_overlapping = overlap.any()
                if ths_in_radius_multiple:
                    cur_paraline_distance_ths = paraline_distance_ths * longer_line.radius
                    cur_contline_distance_ths = contline_distance_ths * longer_line.radius
                else:
                    cur_paraline_distance_ths = paraline_distance_ths
                    cur_contline_distance_ths = contline_distance_ths
                if has_overlapping: #overlapping
                    full_overlap = np.array((overlap[:2].all(), overlap[-2:].all()))
                    if full_overlap.any(): #short line fullly overlap with long line, check min short line endpt to long line
                        short_line = cur_line if full_overlap[1] else check_line
                        long_line = check_line if short_line==cur_line else cur_line
                        _,skew_line_nearest_point_on_short_line = nearest_points(long_line.direction,short_line.direction,long_line.end_points[0],short_line.end_points[0])
                        if not pointline_overlap(short_line.end_points[0],short_line.end_points[1],skew_line_nearest_point_on_short_line):
                            distance = min(point2line_distance(long_line.end_points[0],long_line.end_points[1],short_line.end_points[0]),point2line_distance(long_line.end_points[0],long_line.end_points[1],short_line.end_points[1]))
                        else:
                            distance = point2line_distance(long_line.end_points[0],long_line.end_points[1],skew_line_nearest_point_on_short_line)

                    else: # partial overlap, check nearest endpoint distance
                        # nearest_pointspair = get_nearest_endpoints_pair_non_overlap(cur_line, check_line)
                        # distance = max(point2line_distance(a,b,check_line.end_points[nearest_pointspair[1]]),point2line_distance(c,d, cur_line.end_points[nearest_pointspair[0]]))
                        distance = min(np.array([point2line_distance(a,b,c),point2line_distance(a,b,d),point2line_distance(c,d,a),point2line_distance(c,d,b)])[overlap])

                    if (distance <= cur_paraline_distance_ths):
                        if radius_difference <=0 or (distance < max(cur_paraline_distance_ths*0.5,ransac_residual_ths) and angle < angle_ths*0.5).all(): #not check radius if distance or angle is very small
                            mergelist[cur_Lid, check_id] = True
                            mergelist[check_id, cur_Lid] = True
                            merged = True
                            log_string(LOG_FOUT,f'{cur_Lid} and {check_id} overlap, direction angle {np.rad2deg(angle[idx]):.5f}, vertical distance {distance:.5f}, ths {cur_paraline_distance_ths:.5f}, merged {merged}')
                            cur_line_merge_list.append(check_id)
                        else :
                            radius_diff = abs(cur_line.radius - check_line.radius)
                            radius_diff_percent = radius_diff / longer_line.radius
                            if radius_diff_percent < radius_difference:
                                mergelist[cur_Lid, check_id] = True
                                mergelist[check_id, cur_Lid] = True
                                merged = True
                                log_string(LOG_FOUT,f'{cur_Lid} and {check_id} overlap, direction angle {np.rad2deg(angle[idx]):.5f}, vertical distance {distance:.5f}, ths {cur_paraline_distance_ths:.5f}, merged {merged}')
                                cur_line_merge_list.append(check_id)
                            else:
                                log_string(LOG_FOUT,f'{cur_Lid} and {check_id} overlap, direction angle {np.rad2deg(angle[idx]):.5f}, vertical distance {distance:.5f}, radius difference {radius_diff:.5f} percentage {radius_diff_percent:.5f} too large, ths {radius_difference:.5f}, merged {merged}')

                    else:
                        log_string(LOG_FOUT,f'{cur_Lid} and {check_id} overlap, direction angle {np.rad2deg(angle[idx]):.5f}, vertical distance {distance:.5f} too large, ths {cur_paraline_distance_ths:.5f}, merged {merged}')

                else: #non-overlapping
                    #check if endpoints vector parallel to line direction
                    nearest_pointspair = get_nearest_endpoints_pair_non_overlap(cur_line, check_line)
                    endpoint_vector_norm = (cur_line.end_points[nearest_pointspair[0]]-check_line.end_points[nearest_pointspair[1]])/np.linalg.norm(cur_line.end_points[nearest_pointspair[0]]-check_line.end_points[nearest_pointspair[1]])
                    ang = np.arccos(np.clip(np.abs(np.dot(endpoint_vector_norm,longer_line.direction)),-1.0,1.0))
                    # if ang <angle_ths: # parallel and along one line

                    distance = point2point_distance(cur_line.end_points[nearest_pointspair[0]],check_line.end_points[nearest_pointspair[1]])
                    if (distance*np.cos(ang) <= cur_contline_distance_ths) and (distance*np.sin(ang) <= cur_paraline_distance_ths): #distance along cur_line direction
                        # check radius (in case of reducer - can't merge)
                        # todo: segment continuous co-axis pipes with different radii
                        if radius_difference <=0 or (distance*np.cos(ang) < max(cur_contline_distance_ths*0.5,ransac_residual_ths) and distance*np.sin(ang) <= max(cur_paraline_distance_ths*0.5,ransac_residual_ths) and angle < angle_ths*0.5).all(): #not check radius if distance or angle is very small
                            mergelist[cur_Lid, check_id] = True
                            mergelist[check_id, cur_Lid] = True
                            merged = True
                            log_string(LOG_FOUT,f'{cur_Lid} and {check_id} not overlap, direction angel {np.rad2deg(angle[idx]):.5f}, endpoint vector angle {np.rad2deg(ang):.5f} cont_line dist {distance*np.cos(ang):.5f} paral_line dist {distance*np.sin(ang):.5f}, merged {merged}')
                            cur_line_merge_list.append(check_id)
                        else:
                            radius_diff = abs(cur_line.radius - check_line.radius)
                            radius_diff_percent = radius_diff / longer_line.radius
                            if radius_diff_percent > radius_difference:
                                log_string(LOG_FOUT,f'{cur_Lid} and {check_id} not overlap, direction angel {np.rad2deg(angle[idx]):.2f}, endpoint vector angle {np.rad2deg(ang):.2f} cont_line dist {distance * np.cos(ang):.2f} paral_line dist {distance * np.sin(ang):.2f} radius difference {abs(cur_line.radius - check_line.radius):.2f} percentage {abs(cur_line.radius - check_line.radius) / cur_line.radius:.2f} too large ths {radius_difference}, merged {merged}')
                            else:
                                mergelist[cur_Lid, check_id] = True
                                mergelist[check_id,cur_Lid] = True
                                merged = True
                                log_string(LOG_FOUT,f'{cur_Lid} and {check_id} not overlap, direction angel {np.rad2deg(angle[idx]):.5f}, endpoint vector angle {np.rad2deg(ang):.5f} cont_line dist {distance*np.cos(ang):.5f} paral_line dist {distance*np.sin(ang):.5f}, merged {merged}')
                                cur_line_merge_list.append(check_id)

                    else:
                        log_string(LOG_FOUT,f'{cur_Lid} and {check_id} not overlap, direction angel {np.rad2deg(angle[idx]):.5f}, endpoint vector angle {np.rad2deg(ang):.5f} cont_line dist {distance*np.cos(ang):.5f} paral_line dist {distance*np.sin(ang):.5f} too large, ths {cur_contline_distance_ths:.5f}&{cur_paraline_distance_ths:.5f} merged {merged}')
                    # else:
                    #     print(' %d and %d not overlap, endpoint vector angle %.5f too large threshold %.5f, merged %s' % (cur_Lid, check_id, np.rad2deg(ang),angle_ths, merged))

            log_string(LOG_FOUT,f'{cur_Lid} to-merge list: {cur_line_merge_list}')

        if mergelist.any():
            has_merge = True

        merged_idx_list = []
        merged_lines = []
        for cur_Lid in tqdm(range(0,num_line), desc='merge lines'):
        # for cur_Lid in range(0,num_line):
            cur2merge_list = [cur_Lid]
            if cur_Lid not in merged_idx_list:
                if any(mergelist[cur_Lid]):
                    merge_idx = np.array(mergelist[cur_Lid]).nonzero()[0].tolist()  # (n,1)
                    merged_idx_list.extend([i for i in merge_idx if i not in merged_idx_list])
                    cur2merge_list.extend(merge_idx)
                    for i in cur2merge_list:
                        if i != cur_Lid:
                            if mergelist[i].any():
                                merge_idx = np.array(mergelist[i]).nonzero()[0].tolist()
                                merged_idx_list.extend([i for i in merge_idx if i not in merged_idx_list])
                                cur2merge_list.extend([i for i in merge_idx if i not in cur2merge_list])
                    log_string(LOG_FOUT, f'{cur_Lid} merge with {cur2merge_list}, merged line id {len(merged_lines)}')

                    # num_merge = len(cur2merge_list)
                    xyz_idx = []
                    for i in cur2merge_list:
                        xyz_idx.extend(lines[i].inlier)
                        xyz_idx.extend(lines[i].outlier)
                    xyz_idx = np.unique(xyz_idx)
                    # prevent over-merging: check points lie on a single line, keep at most two lines
                    pca = decomposition.PCA(n_components=3)
                    xyz_mean = StandardScaler(with_std=False).fit(xyz_orig[xyz_idx]).mean_
                    translated_xyz = xyz_orig[xyz_idx] - xyz_mean
                    pca_fitted = pca.fit(translated_xyz)
                    pca_dir = pca_fitted.components_[0]
                    p_dist = np.linalg.norm(np.cross(translated_xyz, pca_dir), axis=-1)
                    avg_p_dist = np.average(p_dist)
                    r_mean = np.average(radius_orig[xyz_idx])
                    merge_clusters = []
                    if avg_p_dist > np.clip(r_mean*0.7*max(itr/2.0,1.0),a_min=None,a_max=ransac_residual_ths*2): # more than one line merged todo: 2-cluster clustering
                        #do RANSAC, inlier & neigobor as 2 clusters
                        merge_clusters = []
                        newLine,neighbor = line_ransac(xyz_orig[xyz_idx], ransac_residual_ths,outlier_min_residual,return_outlier=True,transfer_indice=True,xyz_idx=xyz_idx)
                        merge_clusters.append(newLine.inlier)
                        if len(neighbor) > 20:
                            merge_clusters.append(neighbor)
                            log_string(LOG_FOUT,f'avg_p_dist {avg_p_dist} ths {max(r_mean*0.7*max(itr/2.0,1.0),0.02)} over-merged! break to {len(merge_clusters)} clusters')
                    else:merge_clusters.append(xyz_idx)

                    for xyz_idx in merge_clusters:
                        # TODO: averaging lines by weights of inlier density OR LS/LTS？
                        if line_fit_method == 'ransac':
                            newLine = line_ransac(xyz_orig[xyz_idx], ransac_residual_ths,outlier_min_residual,transfer_indice=True,xyz_idx=xyz_idx,assign_radius=True,radius=radius_orig[xyz_idx],min_inlier_density=min_inlier_density)
                            if newLine != None:
                            #     newLine.inlier = xyz_idx[newLine.inlier]
                            #     newLine.outlier = xyz_idx[newLine.outlier]
                            #     newLine.assign_radius(np.average(radius_orig[newLine.inlier]))
                                merged_lines.append(newLine)
                        elif line_fit_method == 'ransac_projection':
                            newLine = line_ransac(xyz_orig[xyz_idx], ransac_residual_ths,outlier_min_residual,get_full_length=True,transfer_indice=True,xyz_idx=xyz_idx,assign_radius=True,radius=radius_orig[xyz_idx],min_inlier_density=min_inlier_density)
                            if newLine != None:
                                # newLine.inlier = xyz_idx[newLine.inlier]
                                # newLine.outlier = xyz_idx[newLine.outlier]
                                # newLine.assign_radius(np.average(radius_orig[newLine.inlier]))
                                merged_lines.append(newLine)
                        elif line_fit_method =='endpoint_fitting':
                            cur_xyz = xyz_orig[xyz_idx]
                            pca = decomposition.PCA(n_components=3)
                            pca_dir = pca.fit(cur_xyz).components_[0]
                            pca_dir = pca_dir/np.linalg.norm(pca_dir)
                            cur_principal_direction = np.argmax(np.abs(pca_dir))
                            zz = cur_xyz[:, cur_principal_direction]
                            endpoint_min = cur_xyz[np.argmin(zz)]
                            endpoint_max = cur_xyz[np.argmax(zz)]
                            length = np.linalg.norm(endpoint_max - endpoint_min) / 0.01
                            inlier_density = cur_xyz.shape[0] / length
                            if min_inlier_density is not None:
                                if inlier_density > min_inlier_density:
                                    newLine = myline(endpoint_min, pca_dir, [endpoint_min, endpoint_max],
                                                     xyz_idx, xyz_idx, inlier_density)
                                    newLine.assign_radius(np.average(radius_orig[xyz_idx]))
                                    merged_lines.append(newLine)
                            else:
                                newLine = myline(endpoint_min, pca_dir, [endpoint_min, endpoint_max],
                                                 xyz_idx, xyz_idx, inlier_density)
                                newLine.assign_radius(np.average(radius_orig[xyz_idx]))
                                merged_lines.append(newLine)
                        elif line_fit_method == 'least_square':
                            # save_ply(xyz_orig[xyz_idx], os.path.join(file_dir, '..', 'cluster.ply'))
                            reg = LinearRegression().fit(xyz_orig[xyz_idx][:,:2], xyz_orig[xyz_idx][:,-1])
                            print(reg.score(xyz_orig[xyz_idx][:,:2], xyz_orig[xyz_idx][:,-1]))
                            print(reg.coef_)
                            print(reg.intercept_)
                            # ply = np.stack([x_line,y_line,z_line],1)
                            # save_ply(ply,os.path.join(file_dir,'..','fitline.ply'))
                        elif line_fit_method == 'weighted_average': #weight = num of inlier
                            num_inlier = np.zeros((len(cur2merge_list,)),dtype=int)
                            dir = np.zeros((len(cur2merge_list),3))
                            for i in cur2merge_list:
                                num_inlier[i] = lines[i].inlier.shape[0]
                                dir[i,:] = lines[i].direction
                            weight = num_inlier/np.sum(num_inlier,axis=0)
                            weighted_dir = np.average(dir,axis=0,weights=weight)
                            # LOG_FOUT.write('weighted_dir ', weighted_dir)

                            xyz_mean = np.average(xyz_orig[xyz_idx],axis=0) #(3,)
                            translated_xyz = xyz_orig[xyz_idx] - xyz_mean
                            p_dist = np.linalg.norm(np.cross(translated_xyz,weighted_dir),axis=-1)
                            avg_p_dist = np.average(p_dist)
                            merge_inlier = np.less(p_dist,avg_p_dist)
                            merge_inlier_indice = xyz_idx[merge_inlier]
                            cur_principal_direction = np.argmax(np.abs(weighted_dir))
                            zz = translated_xyz[merge_inlier][:, cur_principal_direction]
                            inlier_min_point = translated_xyz[merge_inlier][np.argmin(zz)]
                            inlier_max_point = translated_xyz[merge_inlier][np.argmax(zz)]
                            line_min_point = weighted_dir * (
                                np.dot((inlier_min_point), weighted_dir)) + xyz_mean
                            line_max_point =  weighted_dir * (
                                np.dot((inlier_max_point), weighted_dir)) + xyz_mean
                            length = np.linalg.norm(line_max_point - line_min_point) / 0.01
                            inlier_density = merge_inlier_indice.shape[0] / length
                            newLine = myline(line_min_point, weighted_dir, [line_min_point, line_max_point],merge_inlier_indice, xyz_idx, inlier_density)
                            merged_lines.append(newLine)
                            newLine.assign_radius(np.average(radius_orig[merge_inlier_indice]))
                        elif line_fit_method == 'pca':
                            pca = decomposition.PCA(n_components=3)
                            xyz_mean = StandardScaler(with_std=False).fit(xyz_orig[xyz_idx]).mean_
                            translated_xyz = xyz_orig[xyz_idx] - xyz_mean
                            pca_fitted = pca.fit(translated_xyz)
                            weighted_dir = pca_fitted.components_[0]
                            print(weighted_dir)
                            p_dist = np.linalg.norm(np.cross(translated_xyz, weighted_dir), axis=-1)
                            avg_p_dist = np.average(p_dist)
                            merge_inlier = np.less(p_dist, avg_p_dist) # indice to xyz_orig[xyz_idx]
                            merge_inlier_indice = xyz_idx[merge_inlier] # indice to xyz_orig
                            cur_principal_direction = np.argmax(np.abs(weighted_dir))
                            zz = translated_xyz[merge_inlier][:, cur_principal_direction]
                            inlier_min_point = translated_xyz[merge_inlier][np.argmin(zz)]
                            inlier_max_point = translated_xyz[merge_inlier][np.argmax(zz)]
                            line_min_point = weighted_dir * (
                                np.dot((inlier_min_point), weighted_dir)) + xyz_mean
                            line_max_point = weighted_dir * (
                                np.dot((inlier_max_point), weighted_dir)) + xyz_mean
                            length = np.linalg.norm(line_max_point - line_min_point) / 0.01
                            inlier_density = merge_inlier_indice.shape[0] / length
                            newLine = myline(line_min_point, weighted_dir, [line_min_point, line_max_point],
                                             merge_inlier_indice, xyz_idx, inlier_density)
                            merged_lines.append(newLine)
                            newLine.assign_radius(np.average(radius_orig[merge_inlier_indice]))

        lines = [l for i, l in enumerate(lines) if i not in merged_idx_list]
        not_merged_line_num = len(lines)
        log_string(LOG_FOUT,f'number of not merged lines {len(lines)}')
        lines.extend(merged_lines)
        print('')
    log_string(LOG_FOUT,f'\nTotal merge iteration: {itr}')
    print('Merged lines~')
    return lines

def Merge_lines_parallel(lines, xyz_orig, radius_orig, paraline_distance_ths=0.02, contline_distance_ths=0.1, angle_ths =20 / 180 * np.pi, ransac_residual_ths = 0.005, outlier_min_residual=0.015,min_inlier_density=None, line_fit_method='ransac', radius_difference = 0.2, ths_in_radius_multiple = False,log_file=None,max_itr = 5,pool=None):
    '''only check lines with endpoints within some distance
    iterate until no lines are merged or util max_itr'''
    LOG_FOUT = open(log_file,'w+')
    print('Start merge lines')
    has_merge = True
    itr = 1
    while has_merge and itr <=max_itr: # iterate until no lines are merged
        has_merge = False
        log_string(LOG_FOUT,f'\n\n{itr} time merge!!\n')
        itr+=1

        num_line = len(lines)
        endpoint_list = [l.end_points for l in lines]
        endpoint_list = np.vstack(endpoint_list) #(num_lines*2,3)
        radius_list = np.vstack([l.radius for l in lines]) #(num_lines,)

        dir_list = np.vstack([l.direction for l in lines])
        endpoint1_list = endpoint_list[[i for i in range(len(endpoint_list)) if i%2]]
        if pool is None:
            cur_pool = mp.Pool((mp.cpu_count()))
        else:
            cur_pool = pool
        mid_target = cur_pool.starmap(point2lines_vertical_distance,[(dir_list,endpoint1_list,endpoint_list[i]) for i in range(len(endpoint_list))])

        # regions = [np.concatenate((np.where(mid_target[i]<radius_list[i//2]*max(paraline_distance_ths,contline_distance_ths)*2)[0],np.where(mid_target[i+1]<radius_list[i//2]*max(paraline_distance_ths,contline_distance_ths)*2)[0])) for i in range(0,len(mid_target),2)] #list of arrays(N,) of line id, len=num_lines content:line id
        regions = [np.concatenate((np.where(mid_target[i]<radius_list[i//2]*paraline_distance_ths*2)[0],np.where(mid_target[i+1]<radius_list[i//2]*paraline_distance_ths*2)[0])) for i in range(0,len(mid_target),2)] #list of arrays(N,) of line id, len=num_lines content:line id


        check_map = np.zeros((num_line,num_line),dtype=int)
        mergelist = np.zeros((num_line, num_line), dtype=bool)
        for i,region in tqdm(enumerate(regions), desc='check lines merging'):
            cur_line_merge_list = []
            cur_Lid = i

            cur_line = lines[cur_Lid]
            a = cur_line.end_points[0]
            b = cur_line.end_points[1]
            region = np.unique(region)
            region = region[region != cur_Lid]
            #check angle in batch
            check_angle=np.array([lines[i].direction for i in region])
            cross = np.dot(cur_line.direction, np.transpose(np.reshape(check_angle,(-1,3))))
            angle = np.arccos(np.clip(np.abs(cross),-1.0,1.0))
            region = region[angle<angle_ths]
            angle=angle[angle<angle_ths]
            for idx,j in enumerate(region):
                check_id = j.item() #convert to python int type, for json output use
                if check_map[cur_Lid,check_id]:
                    # log_string(LOG_FOUT," line %d and %d is already checked, skip"%(cur_Lid,check_id))
                    continue
                else:
                    check_map[cur_Lid,check_id] = 1
                    check_map[check_id,cur_Lid] = 1
                check_line = lines[check_id]
                longer_line = cur_line if cur_line.get_length()>check_line.get_length() else check_line
                merged = False
                c = check_line.end_points[0]
                d = check_line.end_points[1]
                # overlap = (pointline_overlap(a, b, c)) | (pointline_overlap(a, b, d))| (pointline_overlap(c,d,a))| (pointline_overlap(c, d, b))
                overlap = np.array((pointline_overlap(a, b, c), pointline_overlap(a, b, d), pointline_overlap(c,d,a),pointline_overlap(c, d, b)))
                has_overlapping = overlap.any()
                if ths_in_radius_multiple:
                    cur_paraline_distance_ths = paraline_distance_ths * longer_line.radius
                    cur_contline_distance_ths = contline_distance_ths * longer_line.radius
                else:
                    cur_paraline_distance_ths = paraline_distance_ths
                    cur_contline_distance_ths = contline_distance_ths
                if has_overlapping: #overlapping
                    full_overlap = np.array((overlap[:2].all(), overlap[-2:].all()))
                    if full_overlap.any(): #short line fullly overlap with long line, check min short line endpt to long line
                        short_line = cur_line if full_overlap[1] else check_line
                        long_line = check_line if short_line==cur_line else cur_line
                        _,skew_line_nearest_point_on_short_line = nearest_points(long_line.direction,short_line.direction,long_line.end_points[0],short_line.end_points[0])
                        if not pointline_overlap(short_line.end_points[0],short_line.end_points[1],skew_line_nearest_point_on_short_line):
                            distance = min(point2line_distance(long_line.end_points[0],long_line.end_points[1],short_line.end_points[0]),point2line_distance(long_line.end_points[0],long_line.end_points[1],short_line.end_points[1]))
                        else:
                            distance = point2line_distance(long_line.end_points[0],long_line.end_points[1],skew_line_nearest_point_on_short_line)

                    else: # partial overlap, check nearest endpoint distance
                        # nearest_pointspair = get_nearest_endpoints_pair_non_overlap(cur_line, check_line)
                        # distance = max(point2line_distance(a,b,check_line.end_points[nearest_pointspair[1]]),point2line_distance(c,d, cur_line.end_points[nearest_pointspair[0]]))
                        distance = min(np.array([point2line_distance(a,b,c),point2line_distance(a,b,d),point2line_distance(c,d,a),point2line_distance(c,d,b)])[overlap])

                    if (distance <= cur_paraline_distance_ths):
                        if radius_difference <=0 or (distance < max(cur_paraline_distance_ths*0.5,ransac_residual_ths) and angle < angle_ths*0.5).all(): #not check radius if distance or angle is very small
                            mergelist[cur_Lid, check_id] = True
                            mergelist[check_id, cur_Lid] = True
                            merged = True
                            log_string(LOG_FOUT,f'{cur_Lid} and {check_id} overlap, direction angle {np.rad2deg(angle[idx]):.5f}, vertical distance {distance:.5f}, ths {cur_paraline_distance_ths:.5f}, merged {merged}')
                            cur_line_merge_list.append(check_id)
                        else :
                            radius_diff = abs(cur_line.radius - check_line.radius)
                            radius_diff_percent = radius_diff / longer_line.radius
                            if radius_diff_percent < radius_difference:
                                mergelist[cur_Lid, check_id] = True
                                mergelist[check_id, cur_Lid] = True
                                merged = True
                                log_string(LOG_FOUT,f'{cur_Lid} and {check_id} overlap, direction angle {np.rad2deg(angle[idx]):.5f}, vertical distance {distance:.5f}, ths {cur_paraline_distance_ths:.5f}, merged {merged}')
                                cur_line_merge_list.append(check_id)
                            else:
                                log_string(LOG_FOUT,f'{cur_Lid} and {check_id} overlap, direction angle {np.rad2deg(angle[idx]):.5f}, vertical distance {distance:.5f}, radius difference {radius_diff:.5f} percentage {radius_diff_percent:.5f} too large, ths {radius_difference:.5f}, merged {merged}')

                    else:
                        log_string(LOG_FOUT,f'{cur_Lid} and {check_id} overlap, direction angle {np.rad2deg(angle[idx]):.5f}, vertical distance {distance:.5f} too large, ths {cur_paraline_distance_ths:.5f}, merged {merged}')

                else: #non-overlapping
                    #check if endpoints vector parallel to line direction
                    nearest_pointspair = get_nearest_endpoints_pair_non_overlap(cur_line, check_line)
                    endpoint_vector_norm = (cur_line.end_points[nearest_pointspair[0]]-check_line.end_points[nearest_pointspair[1]])/np.linalg.norm(cur_line.end_points[nearest_pointspair[0]]-check_line.end_points[nearest_pointspair[1]])
                    ang = np.arccos(np.clip(np.abs(np.dot(endpoint_vector_norm,longer_line.direction)),-1.0,1.0))
                    # if ang <angle_ths: # parallel and along one line

                    distance = point2point_distance(cur_line.end_points[nearest_pointspair[0]],check_line.end_points[nearest_pointspair[1]])
                    if (distance*np.cos(ang) <= cur_contline_distance_ths) and (distance*np.sin(ang) <= cur_paraline_distance_ths): #distance along cur_line direction
                        # check radius (in case of reducer - can't merge)
                        # todo: segment continuous co-axis pipes with different radii
                        if radius_difference <=0 or (distance*np.cos(ang) < max(cur_contline_distance_ths*0.5,ransac_residual_ths) and distance*np.sin(ang) <= max(cur_paraline_distance_ths*0.5,ransac_residual_ths) and angle < angle_ths*0.5).all(): #not check radius if distance or angle is very small
                            mergelist[cur_Lid, check_id] = True
                            mergelist[check_id, cur_Lid] = True
                            merged = True
                            log_string(LOG_FOUT,f'{cur_Lid} and {check_id} not overlap, direction angel {np.rad2deg(angle[idx]):.5f}, endpoint vector angle {np.rad2deg(ang):.5f} cont_line dist {distance*np.cos(ang):.5f} paral_line dist {distance*np.sin(ang):.5f}, merged {merged}')
                            cur_line_merge_list.append(check_id)
                        else:
                            radius_diff = abs(cur_line.radius - check_line.radius)
                            radius_diff_percent = radius_diff / longer_line.radius
                            if radius_diff_percent > radius_difference:
                                log_string(LOG_FOUT,f'{cur_Lid} and {check_id} not overlap, direction angel {np.rad2deg(angle[idx]):.2f}, endpoint vector angle {np.rad2deg(ang):.2f} cont_line dist {distance * np.cos(ang):.2f} paral_line dist {distance * np.sin(ang):.2f} radius difference {abs(cur_line.radius - check_line.radius):.2f} percentage {abs(cur_line.radius - check_line.radius) / cur_line.radius:.2f} too large ths {radius_difference}, merged {merged}')
                            else:
                                mergelist[cur_Lid, check_id] = True
                                mergelist[check_id,cur_Lid] = True
                                merged = True
                                log_string(LOG_FOUT,f'{cur_Lid} and {check_id} not overlap, direction angel {np.rad2deg(angle[idx]):.5f}, endpoint vector angle {np.rad2deg(ang):.5f} cont_line dist {distance*np.cos(ang):.5f} paral_line dist {distance*np.sin(ang):.5f}, merged {merged}')
                                cur_line_merge_list.append(check_id)

                    else:
                        log_string(LOG_FOUT,f'{cur_Lid} and {check_id} not overlap, direction angel {np.rad2deg(angle[idx]):.5f}, endpoint vector angle {np.rad2deg(ang):.5f} cont_line dist {distance*np.cos(ang):.5f} paral_line dist {distance*np.sin(ang):.5f} too large, ths {cur_contline_distance_ths:.5f}&{cur_paraline_distance_ths:.5f} merged {merged}')
                    # else:
                    #     print(' %d and %d not overlap, endpoint vector angle %.5f too large threshold %.5f, merged %s' % (cur_Lid, check_id, np.rad2deg(ang),angle_ths, merged))

            log_string(LOG_FOUT,f'{cur_Lid} to-merge list: {cur_line_merge_list}')


        if mergelist.any():
            has_merge = True

        merged_idx_list = []
        merged_lines = []
        merge_group_list=[]
        for cur_Lid in tqdm(range(0,num_line), desc='group merged lines'):
            cur2merge_list = [cur_Lid]
            if cur_Lid not in merged_idx_list:
                if any(mergelist[cur_Lid]):
                    merge_idx = np.array(mergelist[cur_Lid]).nonzero()[0].tolist()  # (n,1)
                    merged_idx_list.extend([i for i in merge_idx if i not in merged_idx_list])
                    cur2merge_list.extend(merge_idx)
                    for i in cur2merge_list:
                        if i != cur_Lid:
                            if mergelist[i].any():
                                merge_idx = np.array(mergelist[i]).nonzero()[0].tolist()
                                merged_idx_list.extend([i for i in merge_idx if i not in merged_idx_list])
                                cur2merge_list.extend([i for i in merge_idx if i not in cur2merge_list])
                    log_string(LOG_FOUT, f'{cur_Lid} merge with {cur2merge_list}, merged line id {len(merged_lines)}')
                    merge_group_list.append(cur2merge_list)
        if len(merge_group_list)>0:
            merged_lines = cur_pool.starmap(merge_line_ind, [(lines,merge_group_list[i],xyz_orig,radius_orig,ransac_residual_ths,line_fit_method,outlier_min_residual,min_inlier_density,itr) for i in range(len(merge_group_list))])
        if len(merged_lines) > 0:
            merged_lines = np.concatenate(merged_lines).tolist()
            lines = [l for i, l in enumerate(lines) if i not in merged_idx_list]
            not_merged_line_num = len(lines)
            log_string(LOG_FOUT,f'number of not merged lines {not_merged_line_num}')
            lines.extend(merged_lines)
        print('')
    log_string(LOG_FOUT,f'\nTotal merge iteration: {itr}')
    if pool is None:
        cur_pool.close()
        cur_pool.join()
    print('Merged lines~')
    return lines


def merge_line_ind(lines,cur2merge_list,xyz_orig,radius_orig,ransac_residual_ths,line_fit_method,outlier_min_residual=None,min_inlier_density=None,itr=1):
    xyz_idx = []
    merged_lines = []
    for i in cur2merge_list:
        xyz_idx.extend(lines[i].inlier)
        xyz_idx.extend(lines[i].outlier)
    xyz_idx = np.unique(xyz_idx)
    # prevent over-merging: check points lie on a single line, keep at most two lines
    pca = decomposition.PCA(n_components=3)
    xyz_mean = StandardScaler(with_std=False).fit(xyz_orig[xyz_idx]).mean_
    translated_xyz = xyz_orig[xyz_idx] - xyz_mean
    pca_fitted = pca.fit(translated_xyz)
    pca_dir = pca_fitted.components_[0]
    p_dist = np.linalg.norm(np.cross(translated_xyz, pca_dir), axis=-1)
    avg_p_dist = np.average(p_dist)
    r_mean = np.average(radius_orig[xyz_idx])
    merge_clusters = []
    if avg_p_dist > np.clip(r_mean * 0.7 * max(itr / 2.0, 1.0), a_min=None,
                            a_max=ransac_residual_ths * 2):  # more than one line merged todo: 2-cluster clustering
        # do RANSAC, inlier & neigobor as 2 clusters
        merge_clusters = []
        newLine, neighbor = line_ransac(xyz_orig[xyz_idx], ransac_residual_ths, outlier_min_residual,
                                        return_outlier=True, transfer_indice=True, xyz_idx=xyz_idx)
        merge_clusters.append(newLine.inlier)
        if len(neighbor) > 20:
            merge_clusters.append(neighbor)
    else:
        merge_clusters.append(xyz_idx)

    for xyz_idx in merge_clusters:
        if line_fit_method == 'ransac':
            newLine = line_ransac(xyz_orig[xyz_idx], ransac_residual_ths, outlier_min_residual, transfer_indice=True,
                                  xyz_idx=xyz_idx, assign_radius=True, radius=radius_orig[xyz_idx],
                                  min_inlier_density=min_inlier_density)
            if newLine != None:
                #     newLine.inlier = xyz_idx[newLine.inlier]
                #     newLine.outlier = xyz_idx[newLine.outlier]
                #     newLine.assign_radius(np.average(radius_orig[newLine.inlier]))
                merged_lines.append(newLine)
        elif line_fit_method == 'ransac_projection':
            newLine = line_ransac(xyz_orig[xyz_idx], ransac_residual_ths, outlier_min_residual, get_full_length=True,
                                  transfer_indice=True, xyz_idx=xyz_idx, assign_radius=True,
                                  radius=radius_orig[xyz_idx], min_inlier_density=min_inlier_density)
            if newLine != None:
                # newLine.inlier = xyz_idx[newLine.inlier]
                # newLine.outlier = xyz_idx[newLine.outlier]
                # newLine.assign_radius(np.average(radius_orig[newLine.inlier]))
                merged_lines.append(newLine)
        elif line_fit_method == 'endpoint_fitting':
            cur_xyz = xyz_orig[xyz_idx]
            pca = decomposition.PCA(n_components=3)
            pca_dir = pca.fit(cur_xyz).components_[0]
            pca_dir = pca_dir / np.linalg.norm(pca_dir)
            cur_principal_direction = np.argmax(np.abs(pca_dir))
            zz = cur_xyz[:, cur_principal_direction]
            endpoint_min = cur_xyz[np.argmin(zz)]
            endpoint_max = cur_xyz[np.argmax(zz)]
            length = np.linalg.norm(endpoint_max - endpoint_min) / 0.01
            inlier_density = cur_xyz.shape[0] / length
            if min_inlier_density is not None:
                if inlier_density > min_inlier_density:
                    newLine = myline(endpoint_min, pca_dir, [endpoint_min, endpoint_max],
                                     xyz_idx, xyz_idx, inlier_density)
                    newLine.assign_radius(np.average(radius_orig[xyz_idx]))
                    merged_lines.append(newLine)
            else:
                newLine = myline(endpoint_min, pca_dir, [endpoint_min, endpoint_max],
                                 xyz_idx, xyz_idx, inlier_density)
                newLine.assign_radius(np.average(radius_orig[xyz_idx]))
                merged_lines.append(newLine)
        elif line_fit_method == 'least_square':
            # save_ply(xyz_orig[xyz_idx], os.path.join(file_dir, '..', 'cluster.ply'))
            reg = LinearRegression().fit(xyz_orig[xyz_idx][:, :2], xyz_orig[xyz_idx][:, -1])
            print(reg.score(xyz_orig[xyz_idx][:, :2], xyz_orig[xyz_idx][:, -1]))
            print(reg.coef_)
            print(reg.intercept_)

            # ply = np.stack([x_line,y_line,z_line],1)
            # save_ply(ply,os.path.join(file_dir,'..','fitline.ply'))
        elif line_fit_method == 'weighted_average':  # weight = num of inlier
            num_inlier = np.zeros((len(cur2merge_list, )), dtype=int)
            dir = np.zeros((len(cur2merge_list), 3))
            for i in cur2merge_list:
                num_inlier[i] = lines[i].inlier.shape[0]
                dir[i, :] = lines[i].direction
            weight = num_inlier / np.sum(num_inlier, axis=0)
            weighted_dir = np.average(dir, axis=0, weights=weight)
            # LOG_FOUT.write('weighted_dir ', weighted_dir)

            xyz_mean = np.average(xyz_orig[xyz_idx], axis=0)  # (3,)
            translated_xyz = xyz_orig[xyz_idx] - xyz_mean
            p_dist = np.linalg.norm(np.cross(translated_xyz, weighted_dir), axis=-1)
            avg_p_dist = np.average(p_dist)
            merge_inlier = np.less(p_dist, avg_p_dist)
            merge_inlier_indice = xyz_idx[merge_inlier]
            cur_principal_direction = np.argmax(np.abs(weighted_dir))
            zz = translated_xyz[merge_inlier][:, cur_principal_direction]
            inlier_min_point = translated_xyz[merge_inlier][np.argmin(zz)]
            inlier_max_point = translated_xyz[merge_inlier][np.argmax(zz)]
            line_min_point = weighted_dir * (
                np.dot((inlier_min_point), weighted_dir)) + xyz_mean
            line_max_point = weighted_dir * (
                np.dot((inlier_max_point), weighted_dir)) + xyz_mean
            length = np.linalg.norm(line_max_point - line_min_point) / 0.01
            inlier_density = merge_inlier_indice.shape[0] / length
            newLine = myline(line_min_point, weighted_dir, [line_min_point, line_max_point], merge_inlier_indice,
                             xyz_idx, inlier_density)
            merged_lines.append(newLine)
            newLine.assign_radius(np.average(radius_orig[merge_inlier_indice]))
        elif line_fit_method == 'pca':
            pca = decomposition.PCA(n_components=3)
            xyz_mean = StandardScaler(with_std=False).fit(xyz_orig[xyz_idx]).mean_
            translated_xyz = xyz_orig[xyz_idx] - xyz_mean
            pca_fitted = pca.fit(translated_xyz)
            weighted_dir = pca_fitted.components_[0]
            print(weighted_dir)
            p_dist = np.linalg.norm(np.cross(translated_xyz, weighted_dir), axis=-1)
            avg_p_dist = np.average(p_dist)
            merge_inlier = np.less(p_dist, avg_p_dist)  # indice to xyz_orig[xyz_idx]
            merge_inlier_indice = xyz_idx[merge_inlier]  # indice to xyz_orig
            cur_principal_direction = np.argmax(np.abs(weighted_dir))
            zz = translated_xyz[merge_inlier][:, cur_principal_direction]
            inlier_min_point = translated_xyz[merge_inlier][np.argmin(zz)]
            inlier_max_point = translated_xyz[merge_inlier][np.argmax(zz)]
            line_min_point = weighted_dir * (
                np.dot((inlier_min_point), weighted_dir)) + xyz_mean
            line_max_point = weighted_dir * (
                np.dot((inlier_max_point), weighted_dir)) + xyz_mean
            length = np.linalg.norm(line_max_point - line_min_point) / 0.01
            inlier_density = merge_inlier_indice.shape[0] / length
            newLine = myline(line_min_point, weighted_dir, [line_min_point, line_max_point],
                             merge_inlier_indice, xyz_idx, inlier_density)
            merged_lines.append(newLine)
            newLine.assign_radius(np.average(radius_orig[merge_inlier_indice]))
    return merged_lines




def merge_extended_lines(lines, xyz_orig, radius_orig, pred, paraline_distance_ths=0.02, contline_distance_ths=0.1, angle_ths =20 / 180 * np.pi, ransac_residual_ths = 0.005, outlier_min_residual=0.015,min_inlier_density=None, line_fit_method='ransac', radius_difference = 0.2, ths_in_radius_multiple = False,log_file=None):
    '''only check lines with endpoints within some distance
    iterate until no lines are merged'''
    LOG_FOUT = open(log_file,'w+')
    print('Start merge lines')
    has_merge = True
    itr = 1
    while itr==1: # iterate until no lines are merged
        has_merge = False
        log_string(LOG_FOUT,f'\n\n{itr} time merge!!\n')
        itr+=1

        num_line = len(lines)
        endpoint_list = [l.end_points for l in lines]
        endpoint_list = np.vstack(endpoint_list) #(num_lines*2,3)
        radius_list = np.vstack([l.radius for l in lines]) #(num_lines,)

        dir_list = np.vstack([l.direction for l in lines])
        endpoint1_list = endpoint_list[[i for i in range(len(endpoint_list)) if i%2]]
        pool = mp.Pool((mp.cpu_count()))
        mid_target = pool.starmap(point2lines_vertical_distance,[(dir_list,endpoint1_list,endpoint_list[i]) for i in range(len(endpoint_list))])
        pool.close()
        # regions = [np.concatenate((np.where(mid_target[i]<radius_list[i//2]*max(paraline_distance_ths,contline_distance_ths)*2)[0],np.where(mid_target[i+1]<radius_list[i//2]*max(paraline_distance_ths,contline_distance_ths)*2)[0])) for i in range(0,len(mid_target),2)] #list of arrays(N,) of line id, len=num_lines content:line id
        regions = [np.concatenate((np.where(mid_target[i]<radius_list[i//2]*paraline_distance_ths*2)[0],np.where(mid_target[i+1]<radius_list[i//2]*paraline_distance_ths*2)[0])) for i in range(0,len(mid_target),2)] #list of arrays(N,) of line id, len=num_lines content:line id


        check_map = np.zeros((num_line,num_line),dtype=int)
        mergelist = np.zeros((num_line, num_line), dtype=bool)
        for i,region in tqdm(enumerate(regions), desc='check lines merging'):
            cur_line_merge_list = []
        # for i,region in enumerate(regions):
            cur_Lid = i

            cur_line = lines[cur_Lid]
            a = cur_line.end_points[0]
            b = cur_line.end_points[1]
            region = np.unique(region)
            region = region[region != cur_Lid]
            #check angle in batch
            check_angle=np.array([lines[i].direction for i in region])
            cross = np.dot(cur_line.direction, np.transpose(np.reshape(check_angle,(-1,3))))
            angle = np.arccos(np.clip(np.abs(cross),-1.0,1.0))
            region = region[angle<angle_ths]
            angle=angle[angle<angle_ths]
            for idx,j in enumerate(region):
                check_id = j.item() #convert to python int type, for json output use
                if check_map[cur_Lid,check_id]:
                    # log_string(LOG_FOUT," line %d and %d is already checked, skip"%(cur_Lid,check_id))
                    continue
                else:
                    check_map[cur_Lid,check_id] = 1
                    check_map[check_id,cur_Lid] = 1
                check_line = lines[check_id]
                longer_line = cur_line if cur_line.get_length()>check_line.get_length() else check_line
                merged = False
                c = check_line.end_points[0]
                d = check_line.end_points[1]
                # overlap = (pointline_overlap(a, b, c)) | (pointline_overlap(a, b, d))| (pointline_overlap(c,d,a))| (pointline_overlap(c, d, b))
                overlap = np.array((pointline_overlap(a, b, c), pointline_overlap(a, b, d), pointline_overlap(c,d,a),pointline_overlap(c, d, b)))
                has_overlapping = overlap.any()
                if ths_in_radius_multiple:
                    cur_paraline_distance_ths = paraline_distance_ths * longer_line.radius
                    cur_contline_distance_ths = contline_distance_ths * longer_line.radius
                else:
                    cur_paraline_distance_ths = paraline_distance_ths
                    cur_contline_distance_ths = contline_distance_ths
                if has_overlapping: #overlapping
                    full_overlap = np.array((overlap[:2].all(), overlap[-2:].all()))
                    if full_overlap.any(): #short line fullly overlap with long line, check min short line endpt to long line
                        short_line = cur_line if full_overlap[1] else check_line
                        long_line = check_line if short_line==cur_line else cur_line
                        _,skew_line_nearest_point_on_short_line = nearest_points(long_line.direction,short_line.direction,long_line.end_points[0],short_line.end_points[0])
                        if not pointline_overlap(short_line.end_points[0],short_line.end_points[1],skew_line_nearest_point_on_short_line):
                            distance = min(point2line_distance(long_line.end_points[0],long_line.end_points[1],short_line.end_points[0]),point2line_distance(long_line.end_points[0],long_line.end_points[1],short_line.end_points[1]))
                        else:
                            distance = point2line_distance(long_line.end_points[0],long_line.end_points[1],skew_line_nearest_point_on_short_line)

                    else: # partial overlap, check nearest endpoint distance
                        # nearest_pointspair = get_nearest_endpoints_pair_non_overlap(cur_line, check_line)
                        # distance = max(point2line_distance(a,b,check_line.end_points[nearest_pointspair[1]]),point2line_distance(c,d, cur_line.end_points[nearest_pointspair[0]]))
                        distance = min(np.array([point2line_distance(a,b,c),point2line_distance(a,b,d),point2line_distance(c,d,a),point2line_distance(c,d,b)])[overlap])

                    if (distance <= cur_paraline_distance_ths):
                        if radius_difference <=0 or (distance < max(cur_paraline_distance_ths*0.5,ransac_residual_ths) and angle < angle_ths*0.5).all(): #not check radius if distance or angle is very small
                            mergelist[cur_Lid, check_id] = True
                            mergelist[check_id, cur_Lid] = True
                            merged = True
                            log_string(LOG_FOUT,f'{cur_Lid} and {check_id} overlap, direction angle {np.rad2deg(angle[idx]):.5f}, vertical distance {distance:.5f}, ths {cur_paraline_distance_ths:.5f}, merged {merged}')
                            cur_line_merge_list.append(check_id)
                        else :
                            radius_diff = abs(cur_line.radius - check_line.radius)
                            radius_diff_percent = radius_diff / longer_line.radius
                            if radius_diff_percent < radius_difference:
                                mergelist[cur_Lid, check_id] = True
                                mergelist[check_id, cur_Lid] = True
                                merged = True
                                log_string(LOG_FOUT,f'{cur_Lid} and {check_id} overlap, direction angle {np.rad2deg(angle[idx]):.5f}, vertical distance {distance:.5f}, ths {cur_paraline_distance_ths:.5f}, merged {merged}')
                                cur_line_merge_list.append(check_id)
                            else:
                                log_string(LOG_FOUT,f'{cur_Lid} and {check_id} overlap, direction angle {np.rad2deg(angle[idx]):.5f}, vertical distance {distance:.5f}, radius difference {radius_diff:.5f} percentage {radius_diff_percent:.5f} too large, ths {radius_difference:.5f}, merged {merged}')

                    else:
                        log_string(LOG_FOUT,f'{cur_Lid} and {check_id} overlap, direction angle {np.rad2deg(angle[idx]):.5f}, vertical distance {distance:.5f} too large, ths {cur_paraline_distance_ths:.5f}, merged {merged}')

                else: #non-overlapping
                    #check if endpoints vector parallel to line direction
                    nearest_pointspair = get_nearest_endpoints_pair_non_overlap(cur_line, check_line)
                    endpoint_vector_norm = (cur_line.end_points[nearest_pointspair[0]]-check_line.end_points[nearest_pointspair[1]])/np.linalg.norm(cur_line.end_points[nearest_pointspair[0]]-check_line.end_points[nearest_pointspair[1]])
                    ang = np.arccos(np.clip(np.abs(np.dot(endpoint_vector_norm,longer_line.direction)),-1.0,1.0))
                    # if ang <angle_ths: # parallel and along one line

                    distance = point2point_distance(cur_line.end_points[nearest_pointspair[0]],check_line.end_points[nearest_pointspair[1]])
                    if (distance*np.cos(ang) <= cur_contline_distance_ths) and (distance*np.sin(ang) <= cur_paraline_distance_ths): #distance along cur_line direction
                        # check radius (in case of reducer - can't merge)
                        # todo: segment continuous co-axis pipes with different radii
                        if radius_difference <=0 or (distance*np.cos(ang) < max(cur_contline_distance_ths*0.5,ransac_residual_ths) and distance*np.sin(ang) <= max(cur_paraline_distance_ths*0.5,ransac_residual_ths) and angle < angle_ths*0.5).all(): #not check radius if distance or angle is very small
                            mergelist[cur_Lid, check_id] = True
                            mergelist[check_id, cur_Lid] = True
                            merged = True
                            log_string(LOG_FOUT,f'{cur_Lid} and {check_id} not overlap, direction angel {np.rad2deg(angle[idx]):.5f}, endpoint vector angle {np.rad2deg(ang):.5f} cont_line dist {distance*np.cos(ang):.5f} paral_line dist {distance*np.sin(ang):.5f}, merged {merged}')
                            cur_line_merge_list.append(check_id)
                        else:
                            radius_diff = abs(cur_line.radius - check_line.radius)
                            radius_diff_percent = radius_diff / longer_line.radius
                            if radius_diff_percent > radius_difference:
                                log_string(LOG_FOUT,f'{cur_Lid} and {check_id} not overlap, direction angel {np.rad2deg(angle[idx]):.2f}, endpoint vector angle {np.rad2deg(ang):.2f} cont_line dist {distance * np.cos(ang):.2f} paral_line dist {distance * np.sin(ang):.2f} radius difference {abs(cur_line.radius - check_line.radius):.2f} percentage {abs(cur_line.radius - check_line.radius) / cur_line.radius:.2f} too large ths {radius_difference}, merged {merged}')
                            else:
                                mergelist[cur_Lid, check_id] = True
                                mergelist[check_id,cur_Lid] = True
                                merged = True
                                log_string(LOG_FOUT,f'{cur_Lid} and {check_id} not overlap, direction angel {np.rad2deg(angle[idx]):.5f}, endpoint vector angle {np.rad2deg(ang):.5f} cont_line dist {distance*np.cos(ang):.5f} paral_line dist {distance*np.sin(ang):.5f}, merged {merged}')
                                cur_line_merge_list.append(check_id)

                    else:
                        log_string(LOG_FOUT,f'{cur_Lid} and {check_id} not overlap, direction angel {np.rad2deg(angle[idx]):.5f}, endpoint vector angle {np.rad2deg(ang):.5f} cont_line dist {distance*np.cos(ang):.5f} paral_line dist {distance*np.sin(ang):.5f} too large, ths {cur_contline_distance_ths:.5f}&{cur_paraline_distance_ths:.5f} merged {merged}')
                    # else:
                    #     print(' %d and %d not overlap, endpoint vector angle %.5f too large threshold %.5f, merged %s' % (cur_Lid, check_id, np.rad2deg(ang),angle_ths, merged))

            log_string(LOG_FOUT,f'{cur_Lid} to-merge list: {cur_line_merge_list}')

        if mergelist.any():
            has_merge = True

        merged_idx_list = []
        merged_lines = []
        for cur_Lid in tqdm(range(0,num_line), desc='merge lines'):
        # for cur_Lid in range(0,num_line):
            cur2merge_list = [cur_Lid]
            if cur_Lid not in merged_idx_list:
                if any(mergelist[cur_Lid]):
                    merge_idx = np.array(mergelist[cur_Lid]).nonzero()[0].tolist()  # (n,1)
                    merged_idx_list.extend([i for i in merge_idx if i not in merged_idx_list])
                    cur2merge_list.extend(merge_idx)
                    for i in cur2merge_list:
                        if i != cur_Lid:
                            if mergelist[i].any():
                                merge_idx = np.array(mergelist[i]).nonzero()[0].tolist()
                                merged_idx_list.extend([i for i in merge_idx if i not in merged_idx_list])
                                cur2merge_list.extend([i for i in merge_idx if i not in cur2merge_list])
                    log_string(LOG_FOUT, f'{cur_Lid} merge with {cur2merge_list}, merged line id {len(merged_lines)}')

                    # num_merge = len(cur2merge_list)
                    xyz_idx = []
                    pred_idx = []
                    for i in cur2merge_list:
                        xyz_idx.extend(lines[i].inlier)
                        xyz_idx.extend(lines[i].outlier)
                        pred_idx.extend(lines[i].inlier_pred)
                    xyz_idx = np.unique(xyz_idx)
                    pred_idx = np.unique(pred_idx)
                    total_xyz_idx = np.concatenate((xyz_idx,pred_idx))
                    total_inlier = np.concatenate((xyz_orig[xyz_idx],pred[pred_idx]),axis=0)
                    # prevent over-merging: check points lie on a single line, keep at most two lines
                    pca = decomposition.PCA(n_components=3)
                    xyz_mean = StandardScaler(with_std=False).fit(total_inlier).mean_
                    translated_xyz = total_inlier - xyz_mean
                    pca_fitted = pca.fit(translated_xyz)
                    pca_dir = pca_fitted.components_[0]
                    p_dist = np.linalg.norm(np.cross(translated_xyz, pca_dir), axis=-1)
                    avg_p_dist = np.average(p_dist)
                    r_mean = np.average(radius_orig[xyz_idx])
                    merge_clusters = []
                    if avg_p_dist > np.clip(r_mean*0.7*max(itr/2.0,1.0),a_min=None,a_max=ransac_residual_ths*2): # more than one line merged todo: 2-cluster clustering
                        #do RANSAC, inlier & neigobor as 2 clusters
                        merge_clusters = []
                        newLine,neighbor = line_ransac(total_inlier, ransac_residual_ths,outlier_min_residual,return_outlier=True,transfer_indice=True,xyz_idx=total_xyz_idx)
                        merge_clusters.append(newLine.inlier)
                        if len(neighbor) > 20:
                            merge_clusters.append(neighbor)
                            log_string(LOG_FOUT,f'avg_p_dist {avg_p_dist} ths {max(r_mean*0.7*max(itr/2.0,1.0),0.02)} over-merged! break to {len(merge_clusters)} clusters')
                    else:merge_clusters.append(total_xyz_idx)

                    for xyz_idx in merge_clusters:
                        # TODO: averaging lines by weights of inlier density OR LS/LTS？
                        if line_fit_method == 'ransac':
                            newLine = line_ransac(total_inlier, ransac_residual_ths,outlier_min_residual,transfer_indice=False,assign_radius=False,min_inlier_density=min_inlier_density)
                            if newLine != None:
                            #     newLine.inlier = xyz_idx[newLine.inlier]
                            #     newLine.outlier = xyz_idx[newLine.outlier]
                                newLine.assign_radius(r_mean)
                                merged_lines.append(newLine)
                        elif line_fit_method == 'ransac_projection':
                            newLine = line_ransac(total_inlier, ransac_residual_ths,outlier_min_residual,get_full_length=True,transfer_indice=False,assign_radius=False,min_inlier_density=min_inlier_density)
                            if newLine != None:
                                # newLine.inlier = xyz_idx[newLine.inlier]
                                # newLine.outlier = xyz_idx[newLine.outlier]
                                newLine.assign_radius(r_mean)
                                merged_lines.append(newLine)
                        elif line_fit_method =='endpoint_fitting':
                            cur_xyz = total_inlier
                            pca = decomposition.PCA(n_components=3)
                            pca_dir = pca.fit(cur_xyz).components_[0]
                            pca_dir = pca_dir/np.linalg.norm(pca_dir)
                            cur_principal_direction = np.argmax(np.abs(pca_dir))
                            zz = cur_xyz[:, cur_principal_direction]
                            endpoint_min = cur_xyz[np.argmin(zz)]
                            endpoint_max = cur_xyz[np.argmax(zz)]
                            length = np.linalg.norm(endpoint_max - endpoint_min) / 0.01
                            inlier_density = cur_xyz.shape[0] / length
                            if min_inlier_density is not None:
                                if inlier_density > min_inlier_density:
                                    newLine = myline(endpoint_min, pca_dir, [endpoint_min, endpoint_max],
                                                     xyz_idx, xyz_idx, inlier_density)
                                    newLine.assign_radius(np.average(radius_orig[xyz_idx]))
                                    merged_lines.append(newLine)
                            else:
                                newLine = myline(endpoint_min, pca_dir, [endpoint_min, endpoint_max],
                                                 xyz_idx, xyz_idx, inlier_density)
                                newLine.assign_radius(np.average(radius_orig[xyz_idx]))
                                merged_lines.append(newLine)
                        elif line_fit_method == 'least_square':
                            # save_ply(xyz_orig[xyz_idx], os.path.join(file_dir, '..', 'cluster.ply'))
                            reg = LinearRegression().fit(total_inlier[:,:2], total_inlier[:,-1])
                            print(reg.score(total_inlier[:,:2], total_inlier[:,-1]))
                            print(reg.coef_)
                            print(reg.intercept_)

                            # ply = np.stack([x_line,y_line,z_line],1)
                            # save_ply(ply,os.path.join(file_dir,'..','fitline.ply'))
                        elif line_fit_method == 'weighted_average': #weight = num of inlier
                            num_inlier = np.zeros((len(cur2merge_list,)),dtype=int)
                            dir = np.zeros((len(cur2merge_list),3))
                            for i in cur2merge_list:
                                num_inlier[i] = lines[i].inlier.shape[0]
                                dir[i,:] = lines[i].direction
                            weight = num_inlier/np.sum(num_inlier,axis=0)
                            weighted_dir = np.average(dir,axis=0,weights=weight)
                            # LOG_FOUT.write('weighted_dir ', weighted_dir)

                            xyz_mean = np.average(total_inlier,axis=0) #(3,)
                            translated_xyz = total_inlier - xyz_mean
                            p_dist = np.linalg.norm(np.cross(translated_xyz,weighted_dir),axis=-1)
                            avg_p_dist = np.average(p_dist)
                            merge_inlier = np.less(p_dist,avg_p_dist)
                            merge_inlier_indice = xyz_idx[merge_inlier]
                            cur_principal_direction = np.argmax(np.abs(weighted_dir))
                            zz = translated_xyz[merge_inlier][:, cur_principal_direction]
                            inlier_min_point = translated_xyz[merge_inlier][np.argmin(zz)]
                            inlier_max_point = translated_xyz[merge_inlier][np.argmax(zz)]
                            line_min_point = weighted_dir * (
                                np.dot((inlier_min_point), weighted_dir)) + xyz_mean
                            line_max_point =  weighted_dir * (
                                np.dot((inlier_max_point), weighted_dir)) + xyz_mean
                            length = np.linalg.norm(line_max_point - line_min_point) / 0.01
                            inlier_density = merge_inlier_indice.shape[0] / length
                            newLine = myline(line_min_point, weighted_dir, [line_min_point, line_max_point],merge_inlier_indice, xyz_idx, inlier_density)
                            merged_lines.append(newLine)
                            newLine.assign_radius(np.average(radius_orig[merge_inlier_indice]))
                        elif line_fit_method == 'pca':
                            pca = decomposition.PCA(n_components=3)
                            xyz_mean = StandardScaler(with_std=False).fit(total_inlier).mean_
                            translated_xyz = total_inlier - xyz_mean
                            pca_fitted = pca.fit(translated_xyz)
                            weighted_dir = pca_fitted.components_[0]
                            print(weighted_dir)
                            p_dist = np.linalg.norm(np.cross(translated_xyz, weighted_dir), axis=-1)
                            avg_p_dist = np.average(p_dist)
                            merge_inlier = np.less(p_dist, avg_p_dist) # indice to xyz_orig[xyz_idx]
                            merge_inlier_indice = xyz_idx[merge_inlier] # indice to xyz_orig
                            cur_principal_direction = np.argmax(np.abs(weighted_dir))
                            zz = translated_xyz[merge_inlier][:, cur_principal_direction]
                            inlier_min_point = translated_xyz[merge_inlier][np.argmin(zz)]
                            inlier_max_point = translated_xyz[merge_inlier][np.argmax(zz)]
                            line_min_point = weighted_dir * (
                                np.dot((inlier_min_point), weighted_dir)) + xyz_mean
                            line_max_point = weighted_dir * (
                                np.dot((inlier_max_point), weighted_dir)) + xyz_mean
                            length = np.linalg.norm(line_max_point - line_min_point) / 0.01
                            inlier_density = merge_inlier_indice.shape[0] / length
                            newLine = myline(line_min_point, weighted_dir, [line_min_point, line_max_point],
                                             merge_inlier_indice, xyz_idx, inlier_density)
                            merged_lines.append(newLine)
                            newLine.assign_radius(np.average(radius_orig[merge_inlier_indice]))

        lines = [l for i, l in enumerate(lines) if i not in merged_idx_list]
        not_merged_line_num = len(lines)
        log_string(LOG_FOUT,f'number of not merged lines {len(lines)}')
        lines.extend(merged_lines)
        print('')
    log_string(LOG_FOUT,f'\nTotal merge iteration: {itr}')
    print('Merged lines~')
    return lines




def merge_lines_NOT_USED(lines, xyz_orig, radius_orig, paraline_distance_ths=0.02, contline_distance_ths=0.1, angle_ths =20 / 180 * np.pi, ransac_residual_ths = 0.005, outlier_min_residual=0.015, line_fit_method='ransac', radius_difference = 0.2, ths_in_radius_multiple = False, log_file=None):
    ''' Merge near-parallel lines with distance threshold. Traverse thru mergelist for consecutively connected lines in actual merge step
    lines: list of myline objects
    distance_ths: in meter
    angle_ths: in radian
    radius_difference: in % of cur_line radius
    ths_in_radius_multiple: if distance threshold in multiple of cur_line radius
    return: list of merged myline objects'''
    LOG_FOUT = open(log_file,'w+')
    print('Start merge lines')
    num_line = len(lines)

    ### check lines to be merged
    mergelist = np.zeros((num_line, num_line), dtype=bool)
    for cur_Lid in tqdm(range(num_line), desc='check lines merging'):
    # for cur_Lid in range(num_line):
        cur_line = lines[cur_Lid]
        a = cur_line.end_points[0]
        b = cur_line.end_points[1]
        for check_id in range(cur_Lid + 1, num_line):
            check_line = lines[check_id]
            longer_line = cur_line if cur_line.get_length()>check_line.get_length() else check_line
            cross = np.dot(cur_line.direction, check_line.direction)
            angle = np.arccos(np.clip(np.abs(cross),-1.0,1.0))
            merged = False
            if angle < angle_ths:
                c = check_line.end_points[0]
                d = check_line.end_points[1]
                overlap = (pointline_overlap(a, b, c)) | (pointline_overlap(a, b, d))| (pointline_overlap(c,d,a))| (pointline_overlap(c, d, b))
                if ths_in_radius_multiple:
                    cur_paraline_distance_ths = paraline_distance_ths * longer_line.radius
                    cur_contline_distance_ths = contline_distance_ths * longer_line.radius
                else:
                    cur_paraline_distance_ths = paraline_distance_ths
                    cur_contline_distance_ths = contline_distance_ths
                if overlap: #overlapping
                    distance = min(point2line_distance(a, b, c),point2line_distance(a, b, d),point2line_distance(c, d,a),point2line_distance(c, d,b))

                    if (distance <= cur_paraline_distance_ths):
                        if radius_difference >0: #check radius
                            radius_diff = abs(cur_line.radius - check_line.radius)
                            radius_diff_percent = radius_diff / cur_line.radius
                            if radius_diff_percent < radius_difference:
                                mergelist[cur_Lid, check_id] = True
                                mergelist[check_id, cur_Lid] = True
                                merged = True
                                log_string(LOG_FOUT,' %d and %d angle %f overlap, dist %.5f, merged %r' % (cur_Lid, check_id,np.rad2deg(angle), distance,merged))
                            else:
                                log_string(LOG_FOUT,' %d and %d angle %f overlap, dist %.5f, radius difference %.5f percentage %.5f too large, merged %r' % (cur_Lid, check_id, np.rad2deg(angle), distance, radius_diff,radius_diff_percent,merged))
                        else:
                            mergelist[cur_Lid, check_id] = True
                            mergelist[check_id, cur_Lid] = True
                            merged = True
                            log_string(LOG_FOUT,' %d and %d angle %f overlap, dist %.5f, merged %r' % (cur_Lid, check_id, np.rad2deg(angle), distance, merged))
                    else:
                        log_string(LOG_FOUT,' %d and %d angle %f overlap, dist %.5f, threshold %.5f too large, merged %r' % (cur_Lid, check_id, np.rad2deg(angle), distance, cur_paraline_distance_ths, merged))

                else: #non-overlapping
                    #check if endpoints vector parallel to line direction
                    nearest_pointspair = get_nearest_endpoints_pair_non_overlap(cur_line, check_line)
                    endpoint_vector_norm = (cur_line.end_points[nearest_pointspair[0]]-check_line.end_points[nearest_pointspair[1]])/np.linalg.norm(cur_line.end_points[nearest_pointspair[0]]-check_line.end_points[nearest_pointspair[1]])
                    ang = np.arccos(np.clip(np.abs(np.dot(endpoint_vector_norm,cur_line.direction)),-1.0,1.0))
                    # if ang <angle_ths: # parallel and along one line

                    distance = point2point_distance(cur_line.end_points[nearest_pointspair[0]],check_line.end_points[nearest_pointspair[1]])
                    if (distance*np.cos(ang) <= cur_contline_distance_ths) and (distance*np.sin(ang) <= cur_paraline_distance_ths): #distance along cur_line direction
                        # check radius (in case of reducer - can't merge)
                        # todo: segment continuous co-axis pipes with different radii
                        if radius_difference >0:
                            radius_diff = abs(cur_line.radius - check_line.radius)
                            radius_diff_percent = radius_diff / cur_line.radius
                            if radius_diff_percent > radius_difference:
                                log_string(LOG_FOUT,' %d and %d not overlap, endpoint vector angle %.5f cont_line dist %.5f paral_line dist %.5f radius difference %.5f percentage %.5f too large, merged %s' % (cur_Lid, check_id, np.rad2deg(ang), distance * np.cos(ang), distance * np.sin(ang), abs(cur_line.radius - check_line.radius), abs(cur_line.radius - check_line.radius) / cur_line.radius, merged))
                            else:
                                mergelist[cur_Lid, check_id] = True
                                mergelist[check_id,cur_Lid] = True
                                merged = True
                                log_string(LOG_FOUT,' %d and %d not overlap, endpoint vector angle %.5f cont_line dist %.5f paral_line dist %.5f, merged %s' % (cur_Lid, check_id, np.rad2deg(ang), distance*np.cos(ang),distance*np.sin(ang), merged))
                        else:
                            mergelist[cur_Lid, check_id] = True
                            mergelist[check_id, cur_Lid] = True
                            merged = True
                            log_string(LOG_FOUT, ' %d and %d not overlap, endpoint vector angle %.5f cont_line dist %.5f paral_line dist %.5f, merged %s' % (cur_Lid, check_id, np.rad2deg(ang), distance * np.cos(ang), distance * np.sin(ang),merged))
                    else:
                        log_string(LOG_FOUT,' %d and %d not overlap, endpoint vector angle %.5f cont_line dist %.5f paral_line dist %.5f too large, merged %s' % (cur_Lid, check_id, np.rad2deg(ang), distance*np.cos(ang),distance*np.sin(ang), merged))
                    # else:
                    #     print(' %d and %d not overlap, endpoint vector angle %.5f too large threshold %.5f, merged %s' % (cur_Lid, check_id, np.rad2deg(ang),angle_ths, merged))
            else:
                log_string(LOG_FOUT,' %d and %d line direction angle %.5f too large, merged %s' % (
                    cur_Lid, check_id, np.rad2deg(angle), merged))

    merged_idx_list = []
    merged_lines = []
    for cur_Lid in tqdm(range(num_line), desc='merge lines'):
    # for cur_Lid in range(0,num_line):
        cur2merge_list = [cur_Lid]
        if cur_Lid not in merged_idx_list:
            if any(mergelist[cur_Lid]):
                merge_idx = np.array(mergelist[cur_Lid]).nonzero()[0].tolist()  # (n,1)
                merged_idx_list.extend([i for i in merge_idx if i not in merged_idx_list])
                cur2merge_list.extend(merge_idx)
                for i in cur2merge_list:
                    if i != cur_Lid:
                        if mergelist[i].any():
                            merge_idx = np.array(mergelist[i]).nonzero()[0].tolist()
                            merged_idx_list.extend([i for i in merge_idx if i not in merged_idx_list])
                            cur2merge_list.extend([i for i in merge_idx if i not in cur2merge_list])
                log_string(LOG_FOUT, '%d merge with '%cur_Lid + ' , '.join('{:d}'.format(k) for k in cur2merge_list))

                # num_merge = len(cur2merge_list)
                xyz_idx = []
                for i in cur2merge_list:
                    xyz_idx.extend(lines[i].inlier)
                    xyz_idx.extend(lines[i].outlier)
                xyz_idx = np.unique(xyz_idx)
                # merge lines by ransac again for merged data points
                # TODO: averaging lines by weights of inlier density OR LS/LTS？
                if line_fit_method == 'ransac':
                    newLine = line_ransac(xyz_orig[xyz_idx], ransac_residual_ths,outlier_min_residual)
                    if newLine != None:
                        newLine.inlier = xyz_idx[newLine.inlier]
                        newLine.outlier = xyz_idx[newLine.outlier]
                        newLine.assign_radius(np.average(radius_orig[newLine.inlier]))
                        merged_lines.append(newLine)
                elif line_fit_method == 'ransac_projection':
                    newLine = line_ransac(xyz_orig[xyz_idx], ransac_residual_ths,outlier_min_residual,get_full_length=True)
                    if newLine != None:
                        newLine.inlier = xyz_idx[newLine.inlier]
                        newLine.outlier = xyz_idx[newLine.outlier]
                        newLine.assign_radius(np.average(radius_orig[newLine.inlier]))
                        merged_lines.append(newLine)
                elif line_fit_method =='endpoint_fitting':
                    cur_xyz = xyz_orig[xyz_idx]
                    pca = decomposition.PCA(n_components=3)
                    pca_dir = pca.fit(cur_xyz).components_[0]
                    pca_dir = pca_dir/np.linalg.norm(pca_dir)
                    cur_principal_direction = np.argmax(np.abs(pca_dir))
                    zz = cur_xyz[:, cur_principal_direction]
                    endpoint_min = cur_xyz[np.argmin(zz)]
                    endpoint_max = cur_xyz[np.argmax(zz)]
                    length = np.linalg.norm(endpoint_max - endpoint_min) / 0.01
                    inlier_density = cur_xyz.shape[0] / length
                    newLine = myline(endpoint_min, pca_dir, [endpoint_min, endpoint_max],
                                     xyz_idx, xyz_idx, inlier_density)
                    merged_lines.append(newLine)
                    newLine.assign_radius(np.average(radius_orig[xyz_idx]))
                elif line_fit_method == 'least_square':
                    # save_ply(xyz_orig[xyz_idx], os.path.join(file_dir, '..', 'cluster.ply'))
                    reg = LinearRegression().fit(xyz_orig[xyz_idx][:,:2], xyz_orig[xyz_idx][:,-1])
                    print(reg.score(xyz_orig[xyz_idx][:,:2], xyz_orig[xyz_idx][:,-1]))
                    print(reg.coef_)
                    print(reg.intercept_)

                    # ply = np.stack([x_line,y_line,z_line],1)
                    # save_ply(ply,os.path.join(file_dir,'..','fitline.ply'))
                elif line_fit_method == 'weighted_average': #weight = num of inlier
                    num_inlier = np.zeros((len(cur2merge_list,)),dtype=int)
                    dir = np.zeros((len(cur2merge_list),3))
                    for i in cur2merge_list:
                        num_inlier[i] = lines[i].inlier.shape[0]
                        dir[i,:] = lines[i].direction
                    weight = num_inlier/np.sum(num_inlier,axis=0)
                    weighted_dir = np.average(dir,axis=0,weights=weight)
                    # LOG_FOUT.write('weighted_dir ', weighted_dir)

                    xyz_mean = np.average(xyz_orig[xyz_idx],axis=0) #(3,)
                    translated_xyz = xyz_orig[xyz_idx] - xyz_mean
                    p_dist = np.linalg.norm(np.cross(translated_xyz,weighted_dir),axis=-1)
                    avg_p_dist = np.average(p_dist)
                    merge_inlier = np.less(p_dist,avg_p_dist)
                    merge_inlier_indice = xyz_idx[merge_inlier]
                    cur_principal_direction = np.argmax(np.abs(weighted_dir))
                    zz = translated_xyz[merge_inlier][:, cur_principal_direction]
                    inlier_min_point = translated_xyz[merge_inlier][np.argmin(zz)]
                    inlier_max_point = translated_xyz[merge_inlier][np.argmax(zz)]
                    line_min_point = weighted_dir * (
                        np.dot((inlier_min_point), weighted_dir)) + xyz_mean
                    line_max_point =  weighted_dir * (
                        np.dot((inlier_max_point), weighted_dir)) + xyz_mean
                    length = np.linalg.norm(line_max_point - line_min_point) / 0.01
                    inlier_density = merge_inlier_indice.shape[0] / length
                    newLine = myline(line_min_point, weighted_dir, [line_min_point, line_max_point],merge_inlier_indice, xyz_idx, inlier_density)
                    merged_lines.append(newLine)
                    newLine.assign_radius(np.average(radius_orig[merge_inlier_indice]))

                elif line_fit_method == 'pca':
                    pca = decomposition.PCA(n_components=3)
                    xyz_mean = StandardScaler(with_std=False).fit(xyz_orig[xyz_idx]).mean_
                    translated_xyz = xyz_orig[xyz_idx] - xyz_mean
                    pca_fitted = pca.fit(translated_xyz)
                    weighted_dir = pca_fitted.components_[0]
                    print(weighted_dir)
                    p_dist = np.linalg.norm(np.cross(translated_xyz, weighted_dir), axis=-1)
                    avg_p_dist = np.average(p_dist)
                    merge_inlier = np.less(p_dist, avg_p_dist) # indice to xyz_orig[xyz_idx]
                    merge_inlier_indice = xyz_idx[merge_inlier] # indice to xyz_orig
                    cur_principal_direction = np.argmax(np.abs(weighted_dir))
                    zz = translated_xyz[merge_inlier][:, cur_principal_direction]
                    inlier_min_point = translated_xyz[merge_inlier][np.argmin(zz)]
                    inlier_max_point = translated_xyz[merge_inlier][np.argmax(zz)]
                    line_min_point = weighted_dir * (
                        np.dot((inlier_min_point), weighted_dir)) + xyz_mean
                    line_max_point = weighted_dir * (
                        np.dot((inlier_max_point), weighted_dir)) + xyz_mean
                    length = np.linalg.norm(line_max_point - line_min_point) / 0.01
                    inlier_density = merge_inlier_indice.shape[0] / length
                    newLine = myline(line_min_point, weighted_dir, [line_min_point, line_max_point],
                                     merge_inlier_indice, xyz_idx, inlier_density)
                    merged_lines.append(newLine)
                    newLine.assign_radius(np.average(radius_orig[merge_inlier_indice]))

    lines = [l for i, l in enumerate(lines) if i not in merged_idx_list]
    log_string(LOG_FOUT,f'number of not merged lines {len(lines)}')
    lines.extend(merged_lines)
    print('Finish merge lines~')
    return lines

def merge_seg_lines(lines, xyz_orig, radius_orig, paraline_distance_ths=0.02, contline_distance_ths=0.1, angle_ths =3.14 / 180 * 10, ransac_residual_ths = 0.005, outlier_min_residual=0.015, line_fit_method='ransac', radius_difference=0.0):
    '''!!!! NOT USED !!!
    Merge near-parallel lines with distance threshold;
    break merge if endpoint fails to merge due to radius difference
    lines: list of myline objects
    distance_ths: in meter
    angle_ths: in radian
    return: list of merged myline objects'''
    num_line = len(lines)

    ### check lines to be merged
    mergelist = np.zeros((num_line, num_line), dtype=bool)
    # checklist = np.zeros(num_line, dtype=bool)
    for cur_Lid in range(num_line):
        a = lines[cur_Lid].end_points[0]
        b = lines[cur_Lid].end_points[1]
        endpoint_availability = {0:True,1:True}
        checkline_endpoint_linkage = np.tile(np.array([-1]),(num_line))
        for check_id in range(cur_Lid + 1, num_line):
            # if checklist[check_id] == False:
            cross = np.dot(lines[cur_Lid].direction, lines[check_id].direction)
            angle = np.arccos(np.clip(np.abs(cross),-1.0,1.0))
            merged = False
            if angle < angle_ths:
                c = lines[check_id].end_points[0]
                d = lines[check_id].end_points[1]
                overlap = np.array([pointline_overlap(a, b, c), pointline_overlap(a, b, d), pointline_overlap(c, d, a),pointline_overlap(c, d, b)])
                overlap_pt = (np.where(overlap == True))[0]  # array(n,)
                if any(overlap) and overlap_pt.shape[0]>=2:  # overlapping
                    # get min distance of overlapping endpoint to the other line
                    dict = {0: [a, b, c], 1: [a, b, d], 2: [c, d, a], 3: [c, d, b]}
                    distance = point2line_distance(dict[overlap_pt[0]][0], dict[overlap_pt[0]][1],
                                                   dict[overlap_pt[0]][2])
                    for ol_id in overlap_pt[1:]:
                        distance = min(distance, point2line_distance(dict[ol_id][0], dict[ol_id][1], dict[ol_id][2]))
                    if (distance <= paraline_distance_ths):
                        if all(overlap_pt == np.array([0,1])) or all(overlap_pt == np.array([2,3])):#fully overlap
                            mergelist[cur_Lid, check_id] = True
                            mergelist[check_id, cur_Lid] = True
                            merged = True
                            print(' %d and %d fully overlap, dist %.5f, merged %r' % ( cur_Lid, check_id, distance, merged))
                        elif (all(overlap_pt == np.array([0, 2])) or all(overlap_pt == np.array([1, 2]))) and endpoint_availability[0]:
                            if radius_difference >0 and abs(lines[cur_Lid].radius - lines[check_id].radius) > radius_difference:
                                endpoint_availability[0] = False
                                print(' %d and %d overlap, dist %.5f, radius difference %.5f too large, merged %r' % (cur_Lid, check_id, distance, abs(lines[cur_Lid].radius - lines[check_id].radius), merged))
                                # check if any previous lines link to this endpoint
                                if any(checkline_endpoint_linkage == 0):
                                    merged_check_id = np.where(checkline_endpoint_linkage == 0)[0]
                                    mergelist[cur_Lid, merged_check_id] = False
                                    mergelist[merged_check_id, cur_Lid] = False
                            else:
                                mergelist[cur_Lid, check_id] = True
                                mergelist[check_id, cur_Lid] = True
                                merged = True
                                checkline_endpoint_linkage[check_id] = 0
                                print(' %d and %d overlap, dist %.5f, radius difference %.5f, merged %r' % (cur_Lid, check_id, distance, abs(lines[cur_Lid].radius - lines[check_id].radius), merged))
                        elif (all(overlap_pt == np.array([0, 3])) or all(overlap_pt == np.array([1, 3]))) and endpoint_availability[1]:
                            if radius_difference >0 and abs(lines[cur_Lid].radius - lines[check_id].radius)  > radius_difference:
                                endpoint_availability[1] = False
                                print(' %d and %d overlap, dist %.5f, radius difference %.5f too large, merged %r' % (cur_Lid, check_id, distance, abs(lines[cur_Lid].radius - lines[check_id].radius), merged))
                                # check if any previous lines link to this endpoint
                                if any(checkline_endpoint_linkage == 1):
                                    merged_check_id = np.where(checkline_endpoint_linkage == 1)[0]
                                    mergelist[cur_Lid, merged_check_id] = False
                                    mergelist[merged_check_id, cur_Lid] = False
                            else:
                                mergelist[cur_Lid, check_id] = True
                                mergelist[check_id, cur_Lid] = True
                                merged = True
                                checkline_endpoint_linkage[check_id] = 1
                                print(' %d and %d overlap, dist %.5f, radius difference %.5f, merged %r' % (cur_Lid, check_id, distance, abs(lines[cur_Lid].radius - lines[check_id].radius), merged))
                        else: #endpoint not available
                            print(' %d and %d not overlap, cur_line endpoints not available, merged %s'%(cur_Lid, check_id, merged))
                else: #non-overlapping
                    #check if endpoints vector parallel to line direction
                    nearest_pointspair = get_nearest_endpoints_pair_non_overlap(lines[cur_Lid], lines[check_id])
                    if endpoint_availability[nearest_pointspair[0]] == True:
                        endpoint_vector_norm = (lines[cur_Lid].end_points[nearest_pointspair[0]]-lines[check_id].end_points[nearest_pointspair[1]])/np.linalg.norm(lines[cur_Lid].end_points[nearest_pointspair[0]]-lines[check_id].end_points[nearest_pointspair[1]])
                        ang = np.arccos(np.clip(np.abs(np.dot(endpoint_vector_norm,lines[cur_Lid].direction)),-1.0,1.0))
                        # print(ang)
                        # if ang <angle_ths: # parallel and along one line
                        distance = point2point_distance(lines[cur_Lid].end_points[nearest_pointspair[0]],lines[check_id].end_points[nearest_pointspair[1]])
                        # paraline_distance = np.min(point2line_distance(a,b,lines[check_id].end_points[nearest_pointspair[1]]),point2line_distance(c,d,lines[cur_Lid].end_points[nearest_pointspair[0]]))
                        # contline_distance = distance*np.
                        if (distance*np.cos(ang) <= contline_distance_ths) and (distance*np.sin(ang) <= paraline_distance_ths): #distance along cur_line direction
                            # check radius (in case of reducer - can't merge)
                            # todo: segment continuous co-axis pipes with different radii
                            if radius_difference >0 and abs(lines[cur_Lid].radius - lines[check_id].radius)  > radius_difference:
                                endpoint_availability[nearest_pointspair[0]] = False
                                print(' %d and %d not overlap, endpoint vector angle %.5f cont_line dist %.5f paral_line dist %.5f radius difference %.5f percentage %.5f too large, merged %s' % (cur_Lid, check_id, np.rad2deg(ang), distance * np.cos(ang), distance * np.sin(ang), abs(lines[cur_Lid].radius - lines[check_id].radius), abs(lines[cur_Lid].radius - lines[check_id].radius), merged))
                                #check if any previous lines link to this endpoint
                                if any(checkline_endpoint_linkage==nearest_pointspair[0]):
                                    merged_check_id =  np.where(checkline_endpoint_linkage==nearest_pointspair[0])[0]
                                    mergelist[cur_Lid, merged_check_id] = False
                                    mergelist[merged_check_id, cur_Lid] = False
                            else:
                                mergelist[cur_Lid, check_id] = True
                                mergelist[check_id,cur_Lid] = True
                                merged = True
                                checkline_endpoint_linkage[check_id] = nearest_pointspair[0]
                                print(' %d and %d not overlap, endpoint vector angle %.5f cont_line dist %.5f paral_line dist %.5f, merged %s' % (cur_Lid, check_id, np.rad2deg(ang), distance*np.cos(ang),distance*np.sin(ang), merged))
                        else:
                            print(' %d and %d not overlap, endpoint vector angle %.5f cont_line dist %.5f paral_line dist %.5f too large, merged %s' % (cur_Lid, check_id, np.rad2deg(ang), distance*np.cos(ang),distance*np.sin(ang), merged))
                        # else:
                        #     print(' %d and %d not overlap, endpoint vector angle %.5f too large threshold %.5f, merged %s' % (cur_Lid, check_id, np.rad2deg(ang), angle_ths, merged))
                    else:
                        print(' %d and %d not overlap, cur_line endpoints not available, merged %s'%(cur_Lid, check_id, merged))
            else:
                print(' %d and %d not overlap, line direction angle %.5f too large, merged %s' % (
                    cur_Lid, check_id, np.rad2deg(angle), merged))

    merged_idx_list = []
    merged_lines = []
    for cur_Lid in range(0,num_line):
        cur2merge_list = [cur_Lid]
        if cur_Lid not in merged_idx_list:
            if any(mergelist[cur_Lid]):
                merge_idx = np.array(mergelist[cur_Lid]).nonzero()[0].tolist()  # (n,1)
                merged_idx_list.extend([i for i in merge_idx if i not in merged_idx_list])
                cur2merge_list.extend(merge_idx)
                for i in cur2merge_list:
                    if i != cur_Lid:
                        if mergelist[i].any():
                            merge_idx = np.array(mergelist[i]).nonzero()[0].tolist()
                            merged_idx_list.extend([i for i in merge_idx if i not in merged_idx_list])
                            cur2merge_list.extend([i for i in merge_idx if i not in cur2merge_list])
                            # print('also merge with ',merge_idx)
                            # print(cur2merge_list)
                print(cur_Lid, ' merge with ', cur2merge_list)

                # num_merge = len(cur2merge_list)
                xyz_idx = []
                for i in cur2merge_list:
                    xyz_idx.extend(list(lines[i].inlier))
                    xyz_idx.extend(list(lines[i].outlier))
                # print(len(xyz_idx),type(xyz_idx))
                # print(xyz_idx)
                xyz_idx = np.unique(np.array(xyz_idx))
                # save_ply(xyz_orig[xyz_idx],os.path.join('D:\XY\OneDrive - Nanyang Technological University/test_results\MEP_indicetest_rg\merged_pts','cluster%d.ply'%len(merged_lines)))
                # merge lines by ransac again for merged data points
                # TODO: averaging lines by weights of inlier density OR LS/LTS？
                if line_fit_method == 'ransac':
                    newLine = line_ransac(xyz_orig[xyz_idx], ransac_residual_ths,outlier_min_residual)
                    if newLine != None:
                        newLine.inlier = xyz_idx[newLine.inlier]
                        newLine.outlier = xyz_idx[newLine.outlier]
                        newLine.assign_radius(np.average(radius_orig[newLine.inlier]))
                    merged_lines.append(newLine)
                    print(len(merged_lines))
                elif line_fit_method == 'weighted_average': #weight = num of inlier
                    num_inlier = np.zeros((len(cur2merge_list,)),dtype=int)
                    dir = np.zeros((len(cur2merge_list),3))
                    for i in cur2merge_list:
                        num_inlier[i] = lines[i].inlier.shape[0]
                        dir[i,:] = lines[i].direction
                    weight = num_inlier/np.sum(num_inlier,axis=0)
                    weighted_dir = np.average(dir,axis=0,weights=weight)
                    print('weighted_dir ', weighted_dir)

                    xyz_mean = np.average(xyz_orig[xyz_idx],axis=0) #(3,)
                    translated_xyz = xyz_orig[xyz_idx] - xyz_mean
                    p_dist = np.linalg.norm(np.cross(translated_xyz,weighted_dir),axis=-1)
                    avg_p_dist = np.average(p_dist)
                    merge_inlier = np.less(p_dist,avg_p_dist)
                    merge_inlier_indice = xyz_idx[merge_inlier]
                    cur_principal_direction = np.argmax(np.abs(weighted_dir))
                    zz = translated_xyz[merge_inlier][:, cur_principal_direction]
                    inlier_min_point = translated_xyz[merge_inlier][np.argmin(zz)]
                    inlier_max_point = translated_xyz[merge_inlier][np.argmax(zz)]
                    line_min_point = weighted_dir * (
                        np.dot((inlier_min_point), weighted_dir)) + xyz_mean
                    line_max_point =  weighted_dir * (
                        np.dot((inlier_max_point), weighted_dir)) + xyz_mean
                    length = np.linalg.norm(line_max_point - line_min_point) / 0.01
                    inlier_density = merge_inlier_indice.shape[0] / length
                    newLine = myline(line_min_point, weighted_dir, [line_min_point, line_max_point],merge_inlier_indice, xyz_idx, inlier_density)
                    merged_lines.append(newLine)
                    newLine.assign_radius(np.average(radius_orig[merge_inlier_indice]))

                elif line_fit_method == 'pca':
                    pca = decomposition.PCA(n_components=3)
                    xyz_mean = StandardScaler(with_std=False).fit(xyz_orig[xyz_idx]).mean_
                    translated_xyz = xyz_orig[xyz_idx] - xyz_mean
                    pca_fitted = pca.fit(translated_xyz)
                    weighted_dir = pca_fitted.components_[0]
                    print(weighted_dir)
                    p_dist = np.linalg.norm(np.cross(translated_xyz, weighted_dir), axis=-1)
                    avg_p_dist = np.average(p_dist)
                    merge_inlier = np.less(p_dist, avg_p_dist) # indice to xyz_orig[xyz_idx]
                    merge_inlier_indice = xyz_idx[merge_inlier] # indice to xyz_orig
                    cur_principal_direction = np.argmax(np.abs(weighted_dir))
                    zz = translated_xyz[merge_inlier][:, cur_principal_direction]
                    inlier_min_point = translated_xyz[merge_inlier][np.argmin(zz)]
                    inlier_max_point = translated_xyz[merge_inlier][np.argmax(zz)]
                    line_min_point = weighted_dir * (
                        np.dot((inlier_min_point), weighted_dir)) + xyz_mean
                    line_max_point = weighted_dir * (
                        np.dot((inlier_max_point), weighted_dir)) + xyz_mean
                    length = np.linalg.norm(line_max_point - line_min_point) / 0.01
                    inlier_density = merge_inlier_indice.shape[0] / length
                    newLine = myline(line_min_point, weighted_dir, [line_min_point, line_max_point],
                                     merge_inlier_indice, xyz_idx, inlier_density)
                    merged_lines.append(newLine)
                    newLine.assign_radius(np.average(radius_orig[merge_inlier_indice]))
                    print(1)

    lines = [l for i, l in enumerate(lines) if i not in merged_idx_list]
    # zz = [i for i in range(num_line) if i not in merged_idx_list]
    lines.extend(merged_lines)

    return lines

def filter_short_lines(lines, length_ths=None, inlier_density_ths=None, radius_ths=None,confi_ths=None,confi_orig=None, ths_in_radius_multiple=True,log_file=None):
    LOG_FOUT = open(log_file,'w+')
    print("Start filter short lines")
    num_lines = len(lines)
    remove_list = []
    for i in range(num_lines):
        keep_check = []

        if radius_ths is not None:
            radius = lines[i].radius
            if radius < radius_ths:
                keep_check.append(False)
                log_string(LOG_FOUT,'line %d radius %.5f < ths %.5f too small'%(i,radius,radius_ths))
            else:
                keep_check.append(True)
                log_string(LOG_FOUT,'line %d radius %.5f > ths %.5f'%(i,radius,radius_ths))

        if ths_in_radius_multiple:
            cur_length_ths = length_ths*lines[i].radius if length_ths is not None else None
        else:
            cur_length_ths = length_ths
        if cur_length_ths is not None:
            length = lines[i].get_length()
            if length < cur_length_ths:
                keep_check.append(False)
                log_string(LOG_FOUT,'line %d length %.5f < ths %.5f too small'%(i,length,cur_length_ths))
            else:
                keep_check.append(True)
                log_string(LOG_FOUT,'line %d length %.5f > ths %.5f'%(i,length,cur_length_ths))

        if inlier_density_ths is not None:
            in_density = lines[i].get_inlier_density()
            if in_density < inlier_density_ths:
                keep_check.append(False)
                log_string(LOG_FOUT,'line %d inlier density %.5f < ths %.5f too small'%(i,in_density,inlier_density_ths))
            else:
                keep_check.append(True)
                log_string(LOG_FOUT,'line %d inlier density %.5f > ths %.5f'%(i,in_density,inlier_density_ths))

        if confi_ths is not None:
            confi = np.median(confi_orig[np.concatenate((lines[i].inlier,lines[i].outlier))])
            if confi > confi_ths:
                keep_check.append(True)
                log_string(LOG_FOUT,'line %d confidence %.5f > ths %.5f'%(i,confi, confi_ths))
            else:
                keep_check.append(False)
                log_string(LOG_FOUT,'line %d confidence %.5f < ths %.5f too small'%(i,confi, confi_ths))

        if not np.array(keep_check).all():
            remove_list.append(i)

    log_string(LOG_FOUT,f'filter lines: {remove_list}')
    lines = [lines[i] for i in range(num_lines) if i not in remove_list]
    print('Filtered short lines~')
    return lines



def get_scan_inbetween_parallel_lines(scan, cur_line, check_line, nearest_pointspair):
    '''return scan between line1 and line2 neareset endpoints
    nearest_points:list, len(2), index of endpoints'''
    p1 = cur_line.end_points[nearest_pointspair[0]]
    p2 = check_line.end_points[nearest_pointspair[1]]
    scan_clip = scan[(points2line_distance(cur_line.end_points[0], cur_line.end_points[1], scan) < (cur_line.radius + check_line.radius) / 2 * 1.1) | (points2line_distance(check_line.end_points[0], check_line.end_points[1], scan) < (cur_line.radius + check_line.radius) / 2 * 1.1)]
    main_dir_axis1 = np.argmax(np.abs(cur_line.direction))
    main_dir_axis2 = np.argmax(np.abs(check_line.direction))

    if len(scan_clip) > 1:
        if main_dir_axis1==main_dir_axis2:
            min1 = min(p1[main_dir_axis1],p2[main_dir_axis1])
            max1 = max(p1[main_dir_axis1],p2[main_dir_axis1])
            cond = (scan_clip[:, main_dir_axis1] <= max1) & (scan_clip[:, main_dir_axis1] >= min1)
        else:
            min1 = min(p1[main_dir_axis1],p2[main_dir_axis1])
            max1 = max(p1[main_dir_axis1],p2[main_dir_axis1])
            min2 = min(p1[main_dir_axis2],p2[main_dir_axis2])
            max2 = max(p1[main_dir_axis2],p2[main_dir_axis2])
            cond = (scan_clip[:, main_dir_axis1] <= max1) & (scan_clip[:, main_dir_axis1] >= min1) & (scan_clip[:, main_dir_axis2] <= max2) & (scan_clip[:, main_dir_axis2] >= min2)
        scan_clip = scan_clip[cond, :]
    else:
        return None
    return scan_clip

def get_pred_inbetween_parallel_lines(pred, cur_line, check_line, nearest_pointspair,dist=0.03):
    '''return scan between line1 and line2 neareset endpoints
    nearest_points:list, len(2), index of endpoints'''
    p1 = cur_line.end_points[nearest_pointspair[0]]
    p2 = check_line.end_points[nearest_pointspair[1]]
    scan_clip = pred[(points2line_distance(cur_line.end_points[0], cur_line.end_points[1], pred) < max(dist, cur_line.radius)) | (points2line_distance(check_line.end_points[0], check_line.end_points[1], pred) < max(dist, check_line.radius))]
    scan_clip = scan_clip[pointsline_overlap(p1,p2,scan_clip)]

    if len(scan_clip) > 1:
        return scan_clip
    else:
        return None




def get_scan_inbetween_endjoint_lines(scan, cur_line, check_line, nearest_pointspair, intersect):
    '''return scan between line1 and line2 to intersect
    nearest_points:list, len(2), index of endpoints'''
    p1 = cur_line.end_points[nearest_pointspair[0]]
    p2 = check_line.end_points[nearest_pointspair[1]]
    scan_clip = scan[(points2line_distance(cur_line.end_points[0], cur_line.end_points[1], scan) < (cur_line.radius + check_line.radius) / 2 * 1.5) | (points2line_distance(check_line.end_points[0], check_line.end_points[1], scan) < (cur_line.radius + check_line.radius) / 2 * 1.5)]
    main_dir_axis1 = np.argmax(np.abs(cur_line.direction))
    main_dir_axis2 = np.argmax(np.abs(check_line.direction))

    if len(scan_clip) > 1:
        min1 = min(p1[main_dir_axis1], (intersect - (p1-intersect) / np.linalg.norm(p1-intersect) * cur_line.radius)[main_dir_axis1])
        max1 = max(p1[main_dir_axis1],intersect[main_dir_axis1])
        vmin1 = min(p1[main_dir_axis2],intersect[main_dir_axis2]) - cur_line.radius
        vmax1 = max(p1[main_dir_axis2],intersect[main_dir_axis2]) + cur_line.radius

        min2 = min(intersect[main_dir_axis2],p2[main_dir_axis2])
        max2 = max(intersect[main_dir_axis2],p2[main_dir_axis2])
        vmin2 = min(intersect[main_dir_axis1],p2[main_dir_axis1]) - check_line.radius
        vmax2 = max(intersect[main_dir_axis1],p2[main_dir_axis1]) + check_line.radius
        cond = ((scan_clip[:, main_dir_axis1] < max1) & (scan_clip[:, main_dir_axis1] > min1) & (scan_clip[:,main_dir_axis2] < vmax1) & (scan_clip[:,main_dir_axis2] > vmin1)) | ((scan_clip[:, main_dir_axis2] <= max2) & (scan_clip[:, main_dir_axis2] >= min2) & (scan_clip[:,main_dir_axis1] < vmax2) & (scan_clip[:,main_dir_axis1] > vmin2))
        scan_clip = scan_clip[cond, :]
    else:
        return None
    return scan_clip

def get_pred_inbetween_endjoint_lines(pred, cur_line, check_line, nearest_pointspair, intersect, dist=0.03, return_distribution=False,return_dict=False,grid_size=0.01):
    '''return scan between line1 and line2 to intersect
    nearest_points:list, len(2), index of endpoints'''
    p1 = cur_line.end_points[nearest_pointspair[0]]
    p2 = check_line.end_points[nearest_pointspair[1]]
    cur_line_scan = pred[(points2line_distance(cur_line.end_points[0], cur_line.end_points[1], pred) < min(dist, cur_line.radius)) & (pointsline_overlap(p1, intersect, pred)) & np.invert(pointsline_overlap(p1, cur_line.end_points[1 - nearest_pointspair[0]], pred))]
    check_line_scan = pred[(points2line_distance(check_line.end_points[0], check_line.end_points[1], pred) < min(dist, check_line.radius)) & (pointsline_overlap(p2, intersect, pred)) & np.invert(pointsline_overlap(p2, check_line.end_points[1 - nearest_pointspair[1]], pred))]

    scan_clip = np.concatenate((cur_line_scan,check_line_scan),0)
    if len(scan_clip) < 1:

        return None
    else:
        output = (scan_clip,)

        if return_dict:
            cur_d,cur_dict = check_scan_distribution(cur_line_scan,p1,intersect,grid_size,return_dict)
            check_d,check_dict = check_scan_distribution(check_line_scan,p2,intersect,grid_size,return_dict)
            output = output + (cur_d,cur_dict,check_d,check_dict,)
        elif return_distribution:
            cur_d = check_scan_distribution(cur_line_scan,p1,intersect,grid_size,return_dict)
            check_d = check_scan_distribution(check_line_scan,p2,intersect,grid_size,return_dict)
            output = output + (cur_d,check_d,)
        else:
            output = scan_clip
        return output


def get_pred_inbetween_endjoint_lines_no_overlap(pred, cur_line, check_line, nearest_pointspair, intersect, dist=0.03,retrun_distribution=False,grid_size=0.003):
    '''return scan between line1 and line2 to intersect, line1&2 nearest endpt no overlap with the other line
    nearest_points:list, len(2), index of endpoints'''
    p1 = cur_line.end_points[nearest_pointspair[0]]
    p2 = check_line.end_points[nearest_pointspair[1]]
    scan_clip = pred[((points2line_distance(cur_line.end_points[0], cur_line.end_points[1], pred) < min(dist, cur_line.radius)) & (pointsline_overlap(p1, intersect, pred))) | ((points2line_distance(check_line.end_points[0], check_line.end_points[1], pred) < min(dist, check_line.radius)) & (pointsline_overlap(p2, intersect, pred)))]

    if retrun_distribution:
        scan_clip1 = scan_clip[pointsline_overlap(p1,intersect,scan_clip)]
        scan_clip2 = scan_clip[pointsline_overlap(p2,intersect,scan_clip)]
        proj1 = points2line_projection((p1-intersect)/np.linalg.norm(p1-intersect),p1,scan_clip1)
        proj2 = points2line_projection((p2-intersect)/np.linalg.norm(p2-intersect),p2,scan_clip2)
        num_points1 = int(np.ceil(np.linalg.norm(p1 - intersect) / grid_size))
        num_points2 =  int(np.ceil(np.linalg.norm(p2 - intersect) / grid_size))
        d1 = np.zeros(num_points1,dtype=int)
        d2 = np.zeros(num_points2,dtype=int)
        for i in range(proj1.shape[0]):
            x = int(proj1[i] // grid_size)
            d1[x]+=1
        for i in range(proj2.shape[0]):
            x = int(proj2[i] // grid_size)
            d2[x]+=1
        d = np.concatenate((d1,d2))

    if len(scan_clip) > 1:
        if retrun_distribution:
            return scan_clip,d
        else:return scan_clip
    else:
        return None,None


def get_pred_inbetween_endjoint_lines_overlap(pred, cur_line, check_line, nearest_pointspair, intersect, dist=0.03):
    '''return scan between line1 and line2 to intersect, line1&2 nearest endpt no overlap with the other line
    nearest_points:list, len(2), index of endpoints'''
    p1 = cur_line.end_points[nearest_pointspair[0]]
    p2 = check_line.end_points[nearest_pointspair[1]]
    if not pointline_overlap(cur_line.end_points[0],cur_line.end_points[1],intersect):
        cond1 = (points2line_distance(cur_line.end_points[0], cur_line.end_points[1], pred) < min(dist, cur_line.radius)) & (pointsline_overlap(p1, intersect, pred))
    else: cond1 = np.zeros(len(pred),dtype=bool)
    if not pointline_overlap(check_line.end_points[0],check_line.end_points[1],intersect):
        cond2 = (points2line_distance(check_line.end_points[0], check_line.end_points[1], pred) < min(dist, cur_line.radius)) & (pointsline_overlap(p2, intersect, pred))
    else: cond2 = np.zeros(len(pred),dtype=bool)
    cond = cond1 | cond2
    scan_clip = pred[cond,:]

    if len(scan_clip) > 1:
        return scan_clip
    else:
        return None


def get_scan_inbetween_midjoint_lines(scan,end_joint_line,end_joint_line_ept, intersect,mid_joint_line):
    ''''return scan between end_joint_line to intersect'''
    scan_clip = scan[points2line_distance(end_joint_line.end_points[0],end_joint_line.end_points[1],scan) < end_joint_line.radius * 1.5]
    main_dir_axis2 = np.argmax(np.abs(end_joint_line.direction))
    p2 = end_joint_line.end_points[end_joint_line_ept]
    if len(scan_clip) > 1:
        ## move intersect to surface of mid-joint-line
        # intersect = intersect + mid_joint_line.radius*(end_joint_line.end_points[end_joint_line_ept]-intersect)/np.linalg.norm(end_joint_line.end_points[end_joint_line_ept]-intersect)
        min2 = min(intersect[main_dir_axis2],p2[main_dir_axis2])
        max2 = max(intersect[main_dir_axis2],p2[main_dir_axis2])
        cond = (scan_clip[:, main_dir_axis2] <= max2) & (scan_clip[:, main_dir_axis2] >= min2)
        scan_clip = scan_clip[cond, :]
    else:
        return None
    return scan_clip


def get_pred_inbetween_midjoint_lines(pred, end_joint_line, end_joint_line_ept, intersect, mid_joint_line,dist=0.03, return_distribution=False,return_dict=False,grid_size=0.01):
    ''''intersect: on end_joint_line
    return scan between end_joint_line to intersect'''
    p2 = end_joint_line.end_points[end_joint_line_ept]
    # intersect = intersect + (p2-intersect)/np.linalg.norm((p2-intersect))*mid_joint_line.radius
    scan_clip = pred[points2line_distance(end_joint_line.end_points[0], end_joint_line.end_points[1], pred) < min(dist, end_joint_line.radius)]
    scan_clip = scan_clip[pointsline_overlap(intersect,p2,scan_clip)]
    if len(scan_clip) < 1:
        return None
    else:
        output = (scan_clip,)
        if return_dict:
            d,dict = check_scan_distribution(scan_clip,end_joint_line.end_points[end_joint_line_ept],intersect,grid_size,return_dict)
            output += (d,dict,)
        elif return_distribution:
            d = check_scan_distribution(scan_clip,end_joint_line.end_points[end_joint_line_ept],intersect,grid_size,return_dict)
            output += (d,)
        else:
            output = scan_clip
        return output



def get_scan_inbetween_intersects(scan, cur_line, check_line, intersect1, intersect2):
    ''''return scan between endpts of 2 lines, intersect1 on cur_line, intersect2 on check_line'''
    scan_clip = scan[points2line_distance(intersect1,intersect2, scan) < (cur_line.radius + check_line.radius) / 2 * 1.5]
    direction = (intersect1 - intersect2)/np.linalg.norm(intersect1-intersect2)
    main_dir_axis = np.argmax(np.abs(direction))

    if len(scan_clip) > 1:
        min2 = min(intersect1[main_dir_axis],intersect2[main_dir_axis])
        max2 = max(intersect1[main_dir_axis],intersect2[main_dir_axis])
        cond = (scan_clip[:, main_dir_axis] <= max2) & (scan_clip[:, main_dir_axis] >= min2)
        scan_clip = scan_clip[cond, :]
    else:
        return None
    return scan_clip

def get_pred_inbetween_intersects(pred, cur_line, check_line, intersect1, intersect2,dist=0.03):
    ''''return scan between endpts of 2 lines, intersect1 on cur_line, intersect2 on check_line'''
    scan_clip = pred[points2line_distance(intersect1, intersect2, pred) < min(dist, max(cur_line.radius, check_line.radius))]
    scan_clip = scan_clip[pointsline_overlap(intersect1,intersect2,scan_clip)]
    # direction = (intersect1 - intersect2)/np.linalg.norm(intersect1-intersect2)
    # main_dir_axis = np.argmax(np.abs(direction))

    if len(scan_clip) > 1:
        return scan_clip
    else:
        return None


def check_scan_distribution(scan_clip,p1,p2,grid_size=0.003,return_dict=False):
    '''check scan clip even-distribution between p1 and p2'''
    if len(scan_clip) == 0:
        if return_dict:
            return np.array([0]),None
        else:return np.array([0])
    proj = abs(points2line_projection((p1-p2)/np.linalg.norm(p1-p2),p1,scan_clip))
    num_points = int(np.ceil(np.linalg.norm(p1 - p2) / grid_size))
    dict = {}
    d = np.zeros(num_points,dtype=int)
    for i in range(proj.shape[0]):
        x = int(proj[i] // grid_size)
        d[x]+=1
        if return_dict:
            try:dict[x].append(i)
            except:
                dict[x] = []
                dict[x].append(i)
    if return_dict:
        return d,dict
    else: return d

def check_scan_distribution_line_ind(scan_clip,cur_line,check_line,nearest_pointspair,grid_size=0.003,return_dict=False,dist=0.03):
    '''check scan clip distribution along cur_line and check_line'''
    cur_line_scan_id = points2line_distance(cur_line.end_points[0], cur_line.end_points[1], scan_clip) < max(dist, cur_line.radius)
    cur_line_scan = scan_clip[cur_line_scan_id]
    check_line_scan = scan_clip[np.invert(cur_line_scan_id)]
    if return_dict:
        cur_d,cur_dict = check_scan_distribution(cur_line_scan,cur_line.end_points[nearest_pointspair[0]],cur_line.end_points[1-nearest_pointspair[0]],grid_size,return_dict)
        check_d,check_dict = check_scan_distribution(check_line_scan,check_line.end_points[nearest_pointspair[1]],check_line.end_points[1-nearest_pointspair[1]],grid_size,return_dict)
    else:
        cur_d = check_scan_distribution(cur_line_scan,cur_line.end_points[nearest_pointspair[0]],cur_line.end_points[1-nearest_pointspair[0]],grid_size,return_dict)
        check_d = check_scan_distribution(check_line_scan,check_line.end_points[nearest_pointspair[1]],check_line.end_points[1-nearest_pointspair[1]],grid_size,return_dict)
    if return_dict:
        return cur_d,check_d,cur_dict,check_dict
    else: return cur_d,check_d

def check_scan_distribution_4extension(scan_clip, line:myline, line_endpt_id, grid_size=0.003, return_dict=False):
    '''check scan clip distribution histogram anchored at specific endpoint'''
    if len(scan_clip) == 0:
        if return_dict:
            return np.array([0]),None
        else:return np.array([0])
    p1 =line.end_points[line_endpt_id]
    p2 =line.end_points[1-line_endpt_id]
    proj = abs(points2line_projection((p1-p2)/np.linalg.norm(p1-p2),p1,scan_clip))
    num_points = int(np.ceil(np.max(proj) / grid_size))
    d = np.zeros(num_points,dtype=int)
    dict = {}
    for i in range(proj.shape[0]):
        x = int(proj[i] // grid_size)
        d[x]+=1
        if return_dict:
            try:dict[x].append(i)
            except:
                dict[x] = []
                dict[x].append(i)
    if return_dict:
        return d,dict
    else: return d


def merge_missing_lines(lines, scan, xyz_orig, radius_orig, angle_ths, search_region, paraline_distance_ths, line_fit_method, ransac_residual_ths = 0.02, outlier_min_residual=None, scan_density_ths=0.5, log_file=None, grid_size=0.02, radius_difference=0.3):
    '''merge line segments that are not merged because misclassification of scan (so that no centroid pred in between)
    lines should be no overlapping, direction angle < angle_ths
    line each endpt only try once'''
    LOG_FOUT = open(log_file,'w+')
    print('Start make-up missing lines')
    num_line = len(lines)
    endpoint_list = [l.end_points for l in lines]
    endpoint_list = np.vstack(endpoint_list) #(num_lines*2,3)
    radius_list = np.vstack([l.radius for l in lines]) #(num_lines,)
    tree = spatial.cKDTree(endpoint_list, 10,balanced_tree=False,compact_nodes=False)
    pool = mp.Pool((mp.cpu_count()))
    orig_target = pool.starmap(tree.query_ball_point,[(np.reshape(endpoint_list[i],(-1,3)),radius_list[(i//2)]*search_region*2) for i in range(num_line*2)])
    pool.close()
    endpt_regions = [np.concatenate((np.array(orig_target[i][0]),np.array(orig_target[i+1][0]))) for i in range(0,len(orig_target),2)] #list of arrays(N,) len=num_lines, content=endpoint_list indice


    dir_list = np.vstack([l.direction for l in lines])
    endpoint1_list = endpoint_list[[i for i in range(len(endpoint_list)) if i%2]]
    pool = mp.Pool((mp.cpu_count()))
    mid_target = pool.starmap(point2lines_vertical_distance,[(dir_list,endpoint1_list,endpoint_list[i]) for i in range(len(endpoint_list))])
    pool.close()
    regions = [np.concatenate((np.where(mid_target[i]<radius_list[i//2]*paraline_distance_ths*3)[0],np.where(mid_target[i+1]<radius_list[i//2]*paraline_distance_ths*3)[0])) for i in range(0,len(mid_target),2)] #list of arrays(N,) of line id, len=num_lines content:line id


    check_map = np.zeros((num_line,num_line),dtype=int)
    mergelist = np.zeros((num_line, num_line), dtype=bool)
    for i,region in tqdm(enumerate(regions), desc='check lines merging'):
        cur_line_merge_list = []
        cur_line_endpt_merged=np.zeros(2,dtype=bool)
        cur_Lid = i
        cur_line = lines[cur_Lid]
        a = cur_line.end_points[0]
        b = cur_line.end_points[1]
        dir1 = cur_line.direction

        endpt_region = endpt_regions[i]
        endpt_region=endpt_region//2
        endpt_region = np.unique(endpt_region)
        endpt_region = endpt_region[endpt_region != cur_Lid]

        region = np.unique(region)
        region = region[region != cur_Lid]

        region = np.intersect1d(region, endpt_region)

        #check angle in batch
        check_angle=np.array([lines[i].direction for i in region])
        cross = np.dot(cur_line.direction, np.transpose(np.reshape(check_angle,(-1,3))))
        angle = np.arccos(np.clip(np.abs(cross),-1.0,1.0))
        region = region[angle<angle_ths]
        angle=angle[angle<angle_ths]

        if len(region) > 1:
            #sort region by distance
            cur_endpoint_list = [lines[l].end_points for l in region]
            cur_endpoint_list = np.vstack(cur_endpoint_list)
            dist_a = abs(points2point_distance(a,cur_endpoint_list))
            dist_b = abs(points2point_distance(b,cur_endpoint_list))
            dista0 = np.reshape(dist_a[::2],(-1,1))
            dista1 = np.reshape(dist_a[1::2],(-1,1))
            distb0 = np.reshape(dist_b[::2],(-1,1))
            distb1 = np.reshape(dist_b[1::2],(-1,1))
            dist = np.min(np.concatenate((dista0,dista1,distb0,distb1),axis=-1),axis=-1)
            sort = np.argsort(dist)
            region = [region[sortid] for sortid in sort]
            angle = [angle[sortid] for sortid in sort]
        for idx,j in enumerate(region):
            if cur_line_endpt_merged.all():
                log_string(LOG_FOUT,f'{cur_Lid} both endpt checked, break')
                break

            check_id = j.item() #convert to python int type, for json output use
            if not check_map[cur_Lid,check_id]:
                check_map[cur_Lid,check_id] = 1
                check_map[check_id,cur_Lid] = 1
            else:continue
            check_line = lines[check_id]
            longer_line = cur_line if cur_line.get_length()>check_line.get_length() else check_line

            c = check_line.end_points[0]
            d = check_line.end_points[1]
            # overlap = (pointline_overlap(a, b, c)) | (pointline_overlap(a, b, d))| (pointline_overlap(c,d,a))| (pointline_overlap(c, d, b))
            overlap = np.array((pointline_overlap(a, b, c), pointline_overlap(a, b, d), pointline_overlap(c,d,a),pointline_overlap(c, d, b)))
            has_overlapping = overlap.any()
            if not has_overlapping:
                nearest_pointspair = get_nearest_endpoints_pair_non_overlap(cur_line, check_line)
                cur_paraline_distance_ths = paraline_distance_ths * (check_line.radius + cur_line.radius) / 2
                # cur_contline_distance_ths = contline_distance_ths * (check_line.radius+cur_line.radius)/2
                endpoint_vector_norm = (cur_line.end_points[nearest_pointspair[0]]-check_line.end_points[nearest_pointspair[1]])/np.linalg.norm(cur_line.end_points[nearest_pointspair[0]]-check_line.end_points[nearest_pointspair[1]])
                ang = np.arccos(np.clip(np.abs(np.dot(endpoint_vector_norm,longer_line.direction)),-1.0,1.0))
                distance = point2point_distance(cur_line.end_points[nearest_pointspair[0]],check_line.end_points[nearest_pointspair[1]])
                # if (distance*np.cos(ang) <= cur_contline_distance_ths) and (distance*np.sin(ang) <= cur_paraline_distance_ths):
                if distance*np.sin(ang) <= cur_paraline_distance_ths:
                    if cur_line_endpt_merged[nearest_pointspair[0]]==True:
                        log_string(LOG_FOUT,f'{cur_Lid} and {check_id} NOT merged, cur line endpt checked already')
                        continue
                    if radius_difference is not None or radius_difference > 0:
                        cur_radius_diff = abs(cur_line.radius - check_line.radius)
                        radius_diff_percent = cur_radius_diff / longer_line.radius
                        if radius_diff_percent > radius_difference:
                            log_string(LOG_FOUT,f'{cur_Lid} and {check_id} NOT merged, radius difference {cur_radius_diff}, precent {radius_diff_percent} > ths {radius_difference}')
                            continue
                    scan_clip = get_pred_inbetween_parallel_lines(scan, cur_line, check_line, nearest_pointspair)
                    cur_line_endpt_merged[nearest_pointspair[0]]=True
                    if scan_clip is not None:
                        #calculate point density
                        density = len(scan_clip)/(point2point_distance(cur_line.end_points[nearest_pointspair[0]],check_line.end_points[nearest_pointspair[1]])*100) #num of pts per cm
                        # cur_density_ths = 2*np.pi*longer_line.radius*scan_angle_coverage/grid_size**2/100 #surface area*min_angle_coverage/grid_area
                        cur_density_ths=scan_density_ths
                        d = check_scan_distribution(scan_clip,cur_line.end_points[nearest_pointspair[0]],check_line.end_points[nearest_pointspair[1]],grid_size=grid_size)
                        if density > cur_density_ths and np.median(d)>0.5:
                            mergelist[cur_Lid, check_id] = True
                            mergelist[check_id,cur_Lid] = True
                            cur_line_merge_list.append(check_id)

                            log_string(LOG_FOUT,f'{cur_Lid} and {check_id} is merged, density {density}, ths {cur_density_ths}, median {np.median(d)},  vertical distance {distance*np.sin(ang)}, ths {cur_paraline_distance_ths} angle {np.rad2deg(angle[idx])}')
                        else:log_string(LOG_FOUT,f'{cur_Lid} and {check_id} NOT merged, density {density} too small, threshold {cur_density_ths}, angle {np.rad2deg(angle[idx])}')
                    else:log_string(LOG_FOUT,f'{cur_Lid} and {check_id} NOT merged, no scan in between, vertical distance {distance*np.sin(ang)}, continuous distance {distance*np.cos(ang)}, angle {np.rad2deg(angle[idx])}')
                else:log_string(LOG_FOUT,f'{cur_Lid} and {check_id} NOT merged, angle {np.rad2deg(angle[idx])}, vertical distance {distance*np.sin(ang)} too large, thsreshold {cur_paraline_distance_ths}')
            else:log_string(LOG_FOUT,f'{cur_Lid} and {check_id} NOT merged, has overlapping')

        log_string(LOG_FOUT,f'{cur_Lid} to-merge list: {cur_line_merge_list}')

    if mergelist.any():
        merged_idx_list = []
        merged_lines = []
        for cur_Lid in tqdm(range(0,num_line), desc='merge lines'):
            # for cur_Lid in range(0,num_line):
            cur2merge_list = [cur_Lid]
            if cur_Lid not in merged_idx_list:
                if any(mergelist[cur_Lid]):
                    merge_idx = np.array(mergelist[cur_Lid]).nonzero()[0].tolist()  # (n,1)
                    merged_idx_list.extend([i for i in merge_idx if i not in merged_idx_list])
                    cur2merge_list.extend(merge_idx)
                    for i in cur2merge_list:
                        if i != cur_Lid:
                            if mergelist[i].any():
                                merge_idx = np.array(mergelist[i]).nonzero()[0].tolist()
                                merged_idx_list.extend([i for i in merge_idx if i not in merged_idx_list])
                                cur2merge_list.extend([i for i in merge_idx if i not in cur2merge_list])
                    log_string(LOG_FOUT, f'{cur_Lid} merge with {cur2merge_list}, merged line id {len(merged_lines)}')

                    # num_merge = len(cur2merge_list)
                    xyz_idx = []
                    for i in cur2merge_list:
                        xyz_idx.extend(lines[i].inlier)
                        xyz_idx.extend(lines[i].outlier)
                    xyz_idx = np.unique(xyz_idx)
                    # prevent over-merging: check points lie on a single line, keep at most two lines
                    pca = decomposition.PCA(n_components=3)
                    xyz_mean = StandardScaler(with_std=False).fit(xyz_orig[xyz_idx]).mean_
                    translated_xyz = xyz_orig[xyz_idx] - xyz_mean
                    pca_fitted = pca.fit(translated_xyz)
                    pca_dir = pca_fitted.components_[0]
                    p_dist = np.linalg.norm(np.cross(translated_xyz, pca_dir), axis=-1)
                    avg_p_dist = np.average(p_dist)
                    r_mean = np.average(radius_orig[xyz_idx])
                    merge_clusters = []
                    if avg_p_dist > max(r_mean*0.7,1): # more than one line merged
                        #do RANSAC, inlier & neigobor as 2 clusters
                        merge_clusters = []
                        newLine,neighbor = line_ransac(xyz_orig[xyz_idx], r_mean,None,return_outlier=True,transfer_indice=True,xyz_idx=xyz_idx)
                        merge_clusters.append(newLine.inlier)
                        if len(neighbor) > 20:
                            merge_clusters.append(neighbor)
                            log_string(LOG_FOUT,f'avg_p_dist {avg_p_dist} r_mean {r_mean*0.7} over-merged! break to {len(merge_clusters)} clusters')
                    else:merge_clusters.append(xyz_idx)

                    for xyz_idx in merge_clusters:
                        if line_fit_method == 'ransac':
                            newLine = line_ransac(xyz_orig[xyz_idx], ransac_residual_ths,None,transfer_indice=True,xyz_idx=xyz_idx,assign_radius=True,radius=radius_orig[xyz_idx])
                            if newLine != None:
                                # total_dist = point2lines_vertical_distance(newLine.direction,newLine.end_points[0],xyz_total)
                                # total_overlap = pointsline_overlap(newLine.end_points[0],newLine.end_points[1],xyz_total)
                                # total_inlier = np.count_nonzero((total_dist<ransac_residual_ths) & total_overlap)
                                # if total_inlier/newLine.get_length() > 0.5:
                                merged_lines.append(newLine)
                        elif line_fit_method == 'ransac_projection':
                            newLine = line_ransac(xyz_orig[xyz_idx], ransac_residual_ths,outlier_min_residual,get_full_length=True,transfer_indice=True,xyz_idx=xyz_idx,assign_radius=True,radius=radius_orig[xyz_idx])
                            if newLine != None:
                                #todo: check inlier density in xyz_total
                                merged_lines.append(newLine)
                        elif line_fit_method =='endpoint_fitting':
                            cur_xyz = xyz_orig[xyz_idx]
                            pca = decomposition.PCA(n_components=3)
                            pca_dir = pca.fit(cur_xyz).components_[0]
                            pca_dir = pca_dir/np.linalg.norm(pca_dir)
                            cur_principal_direction = np.argmax(np.abs(pca_dir))
                            zz = cur_xyz[:, cur_principal_direction]
                            endpoint_min = cur_xyz[np.argmin(zz)]
                            endpoint_max = cur_xyz[np.argmax(zz)]
                            length = np.linalg.norm(endpoint_max - endpoint_min) / 0.01
                            inlier_density = cur_xyz.shape[0] / length
                            newLine = myline(endpoint_min, pca_dir, [endpoint_min, endpoint_max],
                                             xyz_idx, xyz_idx, inlier_density)
                            merged_lines.append(newLine)
                            newLine.assign_radius(np.average(radius_orig[xyz_idx]))
        lines = [l for i, l in enumerate(lines) if i not in merged_idx_list]
        log_string(LOG_FOUT,f'number of not merged lines {len(lines)}')
        lines.extend(merged_lines)
    print('Finished make-up missing lines')
    return lines


def Merge_missing_lines_parallel(lines, scan, xyz_orig, radius_orig, angle_ths, search_region, paraline_distance_ths, line_fit_method, ransac_residual_ths = 0.02, outlier_min_residual=None, scan_density_ths=0.5, log_file=None, grid_size=0.02, radius_difference=0.3,pool=None):
    '''merge line segments that are not merged because misclassification of scan (so that no centroid pred in between)
    lines should be no overlapping, direction angle < angle_ths
    line each endpt only try once'''
    print('Start make-up missing lines')
    num_line = len(lines)
    endpoint_list = [l.end_points for l in lines]
    endpoint_list = np.vstack(endpoint_list) #(num_lines*2,3)
    radius_list = np.vstack([l.radius for l in lines]) #(num_lines,)
    tree = spatial.cKDTree(endpoint_list, 10,balanced_tree=False,compact_nodes=False)
    if pool is None:
        cur_pool = mp.Pool((mp.cpu_count()))
    else:cur_pool = pool
    orig_target = cur_pool.starmap(tree.query_ball_point,[(np.reshape(endpoint_list[i],(-1,3)),radius_list[(i//2)]*search_region*2) for i in range(num_line*2)])
    endpt_regions = [np.concatenate((np.array(orig_target[i][0]),np.array(orig_target[i+1][0]))) for i in range(0,len(orig_target),2)] #list of arrays(N,) len=num_lines, content=endpoint_list indice


    dir_list = np.vstack([l.direction for l in lines])
    endpoint1_list = endpoint_list[[i for i in range(len(endpoint_list)) if i%2]]
    mid_target = cur_pool.starmap(point2lines_vertical_distance,[(dir_list,endpoint1_list,endpoint_list[i]) for i in range(len(endpoint_list))])
    regions = [np.concatenate((np.where(mid_target[i]<radius_list[i//2]*paraline_distance_ths*3)[0],np.where(mid_target[i+1]<radius_list[i//2]*paraline_distance_ths*3)[0])) for i in range(0,len(mid_target),2)] #list of arrays(N,) of line id, len=num_lines content:line id


    endpt_regions = [np.unique(r//2) for r in endpt_regions]
    regions = [np.unique(r) for r in regions]
    regions = [np.intersect1d(regions[i], endpt_regions[i]) for i in range(len(regions))]
    check_map = np.zeros((num_line,num_line),dtype=int)
    for i in range(num_line):
        check_map[i, :][regions[i]] = True
    check_map = (check_map) | (check_map.transpose())
    ltri_id = np.tril_indices(num_line)
    check_map[ltri_id] = False
    mergelist = np.zeros((num_line, num_line), dtype=bool)
    merge_list = cur_pool.starmap(check_merge_missing_line_ind,
                              [(i,lines,check_map[i,:],paraline_distance_ths,radius_difference,scan,scan_density_ths,grid_size,angle_ths,log_file) for i in range(num_line)])

    for i in range(num_line):
        mergelist[i,merge_list[i]] = True
        mergelist[merge_list[i],i] = True
    LOG_FOUT = open(log_file,'a+')

    if mergelist.any():
        merged_idx_list = np.zeros(num_line,dtype=int)
        merged_lines = []
        merge_group_list=[]
        for cur_Lid in tqdm(range(0,num_line), desc='merge lines'):
            cur2merge_list = [cur_Lid]
            if merged_idx_list[cur_Lid]==0:
                if any(mergelist[cur_Lid]):
                    merge_idx = np.array(mergelist[cur_Lid]).nonzero()[0].tolist()  # (n,1)
                    merged_idx_list[merge_idx]=1
                    cur2merge_list.extend(merge_idx)
                    merged_idx_list[cur_Lid] = 1
                    for i in cur2merge_list:
                        if i != cur_Lid:
                            if mergelist[i].any():
                                merge_idx = np.array(mergelist[i]).nonzero()[0].tolist()
                                merged_idx_list[merge_idx]=1
                                cur2merge_list.extend([i for i in merge_idx if i not in cur2merge_list])
                    log_string(LOG_FOUT, f'{cur_Lid} merge with {cur2merge_list}, merged line id {len(merged_lines)}')
                    merge_group_list.append(cur2merge_list)

        merged_lines = cur_pool.starmap(merge_line_ind, [(lines, merge_group_list[i], xyz_orig, radius_orig,ransac_residual_ths, line_fit_method, outlier_min_residual,None, 1) for i in range(len(merge_group_list))])
        if len(merged_lines) > 0:
            merged_lines = np.concatenate(merged_lines).tolist()
            lines = [l for i, l in enumerate(lines) if merged_idx_list[i]==0]
            log_string(LOG_FOUT,f'number of not merged lines {len(lines)}')
            lines.extend(merged_lines)
    if pool is None:
        cur_pool.close()
        cur_pool.join()
    print('Finished make-up missing lines')
    return lines


def check_merge_missing_line_ind(cur_Lid,lines,region,paraline_distance_ths,radius_difference,scan,scan_density_ths,grid_size,angle_ths,log_file=None):
    if log_file is not None:
        LOG_FOUT= open(log_file,'a+')
    else:
        LOG_FOUT= None
    cur_line_merge_list = []
    cur_line_endpt_merged = np.zeros(2, dtype=bool)
    cur_line = lines[cur_Lid]
    a = cur_line.end_points[0]
    b = cur_line.end_points[1]
    dir1 = cur_line.direction

    region = np.where(region==1)[0]

    # check angle in batch
    check_angle = np.array([lines[i].direction for i in region])
    cross = np.dot(cur_line.direction, np.transpose(np.reshape(check_angle, (-1, 3))))
    angle = np.arccos(np.clip(np.abs(cross), -1.0, 1.0))
    region = region[angle < angle_ths]
    angle = angle[angle < angle_ths]

    if len(region) > 1:
        # sort region by distance
        cur_endpoint_list = [lines[l].end_points for l in region]
        cur_endpoint_list = np.vstack(cur_endpoint_list)
        dist_a = abs(points2point_distance(a, cur_endpoint_list))
        dist_b = abs(points2point_distance(b, cur_endpoint_list))
        dista0 = np.reshape(dist_a[::2], (-1, 1))
        dista1 = np.reshape(dist_a[1::2], (-1, 1))
        distb0 = np.reshape(dist_b[::2], (-1, 1))
        distb1 = np.reshape(dist_b[1::2], (-1, 1))
        dist = np.min(np.concatenate((dista0, dista1, distb0, distb1), axis=-1), axis=-1)
        sort = np.argsort(dist)
        region = [region[sortid] for sortid in sort]
        angle = [angle[sortid] for sortid in sort]
    for idx, j in enumerate(region):
        if cur_line_endpt_merged.all():
            break

        check_id = j.item()  # convert to python int type, for json output use
        check_line = lines[check_id]
        longer_line = cur_line if cur_line.get_length() > check_line.get_length() else check_line

        c = check_line.end_points[0]
        d = check_line.end_points[1]
        overlap = np.array((pointline_overlap(a, b, c), pointline_overlap(a, b, d), pointline_overlap(c, d, a),
                            pointline_overlap(c, d, b)))
        has_overlapping = overlap.any()
        if not has_overlapping:
            nearest_pointspair = get_nearest_endpoints_pair_non_overlap(cur_line, check_line)
            cur_paraline_distance_ths = paraline_distance_ths * (check_line.radius + cur_line.radius) / 2
            # cur_contline_distance_ths = contline_distance_ths * (check_line.radius+cur_line.radius)/2
            endpoint_vector_norm = (cur_line.end_points[nearest_pointspair[0]] - check_line.end_points[
                nearest_pointspair[1]]) / np.linalg.norm(
                cur_line.end_points[nearest_pointspair[0]] - check_line.end_points[nearest_pointspair[1]])
            ang = np.arccos(np.clip(np.abs(np.dot(endpoint_vector_norm, longer_line.direction)), -1.0, 1.0))
            distance = point2point_distance(cur_line.end_points[nearest_pointspair[0]],
                                            check_line.end_points[nearest_pointspair[1]])
            # if (distance*np.cos(ang) <= cur_contline_distance_ths) and (distance*np.sin(ang) <= cur_paraline_distance_ths):
            if distance * np.sin(ang) <= cur_paraline_distance_ths:
                if cur_line_endpt_merged[nearest_pointspair[0]] == True:
                    continue
                if radius_difference is not None or radius_difference > 0:
                    cur_radius_diff = abs(cur_line.radius - check_line.radius)
                    radius_diff_percent = cur_radius_diff / longer_line.radius
                    if radius_diff_percent > radius_difference:
                        continue
                scan_clip = get_pred_inbetween_parallel_lines(scan, cur_line, check_line, nearest_pointspair)
                cur_line_endpt_merged[nearest_pointspair[0]] = True
                if scan_clip is not None:
                    # calculate point density
                    density = len(scan_clip) / (point2point_distance(cur_line.end_points[nearest_pointspair[0]], check_line.end_points[nearest_pointspair[1]]) * 100)  # num of pts per cm

                    cur_density_ths = scan_density_ths
                    d = check_scan_distribution(scan_clip, cur_line.end_points[nearest_pointspair[0]],
                                                check_line.end_points[nearest_pointspair[1]], grid_size=grid_size)
                    if density > cur_density_ths and np.median(d) > 0.5:
                        cur_line_merge_list.append(check_id)
                        log_string(LOG_FOUT,f'{cur_Lid} and {check_id} is merged, density {density}, ths {cur_density_ths}, median {np.median(d)},  vertical distance {distance*np.sin(ang)}, ths {cur_paraline_distance_ths} angle {np.rad2deg(angle[idx])}')
                    else:log_string(LOG_FOUT,f'{cur_Lid} and {check_id} NOT merged, density {density} median {np.median(d)} too small, density ths {cur_density_ths}, angle {np.rad2deg(angle[idx])}')
                else:log_string(LOG_FOUT,f'{cur_Lid} and {check_id} NOT merged, no scan in between, vertical distance {distance*np.sin(ang)}, continuous distance {distance*np.cos(ang)}, angle {np.rad2deg(angle[idx])}')
            else:log_string(LOG_FOUT,f'{cur_Lid} and {check_id} NOT merged, angle {np.rad2deg(angle[idx])}, vertical distance {distance*np.sin(ang)} too large, thsreshold {cur_paraline_distance_ths}')
        else:log_string(LOG_FOUT,f'{cur_Lid} and {check_id} NOT merged, has overlapping')
    if LOG_FOUT is not None:
        LOG_FOUT.close()
    return cur_line_merge_list


def extend_lines(lines, pred,xyz_orig,inlier_dist=0.03,region_grow_dist=0.08,free_endpt_only=False):
    num_line = len(lines)
    for line_id in tqdm(range(num_line),desc='extending lines'):
        cur_line = lines[line_id]
        lines[line_id] = extend_line_ind(cur_line,pred,xyz_orig,inlier_dist,region_grow_dist,free_endpt_only)

    return lines

def Extend_lines_parallel(pool,lines, pred,xyz_orig,inlier_dist=0.03,region_grow_dist=0.08,free_endpt_only=False,log_file=None):
    print('start extend lines')
    num_line = len(lines)
    if pool is None:
        pool = mp.Pool((mp.cpu_count()))
        lines = pool.starmap(extend_line_ind,[(i,lines[i],pred,xyz_orig,inlier_dist,region_grow_dist,free_endpt_only,log_file) for i in tqdm(range(num_line),desc='extend lines')])
        pool.close()
        pool.join()
    else:
        lines = pool.starmap(extend_line_ind,[(i,lines[i],pred,xyz_orig,inlier_dist,region_grow_dist,log_file) for i in tqdm(range(num_line),desc='extend lines')])
    return lines

def extend_line_ind(cur_Lid,cur_line,pred,xyz_orig,inlier_dist,region_grow_dist,free_endpt_only=False,log_file=None):
    if log_file is not None:
        LOG_FOUT= open(log_file,'a+')
    else:
        LOG_FOUT= None
    if cur_line.radius < 0:
        return cur_line
    dist = point2lines_vertical_distance(cur_line.direction, cur_line.end_points[0], pred)
    full_inlier_id = dist<inlier_dist
    cur_pred_all = pred[full_inlier_id]
    cur_pred_all_id2pred = np.where(full_inlier_id)[0]
    if not free_endpt_only:
        cur_pred0 = cur_pred_all[pointsline_overlap(cur_line.end_points[0],cur_line.end_points[1],cur_pred_all,return_t=True)[1]<0]
        cur_pred0_id2pred = cur_pred_all_id2pred[pointsline_overlap(cur_line.end_points[0],cur_line.end_points[1],cur_pred_all,return_t=True)[1]<0]
        cur_pred1 = cur_pred_all[pointsline_overlap(cur_line.end_points[0],cur_line.end_points[1],cur_pred_all,return_t=True)[1]>1]
        cur_pred1_id2pred = cur_pred_all_id2pred[pointsline_overlap(cur_line.end_points[0],cur_line.end_points[1],cur_pred_all,return_t=True)[1]>1]
        cur_pred_list = [cur_pred0,cur_pred1]
        cur_pred_id_list = [cur_pred0_id2pred,cur_pred1_id2pred]
    else:
        endpt_id_list = [i for i in range(2) if len(cur_line.intersection['end'][i])==0]
        if len(endpt_id_list) == 0:
            return cur_line
        else:
            cur_pred_list = []
            cur_pred_id_list = []
            for endpt_id in endpt_id_list:
                _,t=pointsline_overlap(cur_line.end_points[0],cur_line.end_points[1],cur_line.end_points[endpt_id],return_t=True)
                if abs(t)<0.01:
                    cur_pred_list.append(cur_pred_all[pointsline_overlap(cur_line.end_points[0],cur_line.end_points[1],cur_pred_all,return_t=True)[1]<0])
                    cur_pred_id_list.append(cur_pred_all_id2pred[pointsline_overlap(cur_line.end_points[0],cur_line.end_points[1],cur_pred_all,return_t=True)[1]<0])
                elif abs(t) -1 < 0.01:
                    cur_pred_list.append(cur_pred_all[pointsline_overlap(cur_line.end_points[0],cur_line.end_points[1],cur_pred_all,return_t=True)[1]>1])
                    cur_pred_id_list.append(cur_pred_all_id2pred[pointsline_overlap(cur_line.end_points[0],cur_line.end_points[1],cur_pred_all,return_t=True)[1]>1])
                else: raise ValueError
    segmented_regions = []
    for cpid,cur_pred in enumerate(cur_pred_list):
        if len(cur_pred) < 5:
            continue
        start_pt = np.argmin(points2point_distance(cur_line.end_points[0],cur_pred))
        endpt = 0 if point2point_distance(cur_pred[start_pt],cur_line.end_points[0])<point2point_distance(cur_pred[start_pt],cur_line.end_points[1]) else 1
        d,dict= check_scan_distribution_4extension(cur_pred, cur_line, endpt, grid_size=region_grow_dist, return_dict=True)

        zero_bins = np.where(d==0)[0]
        if len(zero_bins) == 0:
            stop_id = len(d) + 1
        else:
            stop_id = zero_bins[0]
            for bin_id in zero_bins:
                if bin_id+1 in zero_bins:
                    stop_id = bin_id
                    break
        cur_region = []
        for jj in range(stop_id):
            try:cur_region.extend(dict[jj])
            except KeyError: pass
        if len(cur_region)>3:
            segmented_regions.extend(cur_pred_id_list[cpid][cur_region])
    if len(segmented_regions) > 0:
        full_inlier = np.concatenate((xyz_orig[cur_line.inlier],pred[segmented_regions]),axis=0)
        newLine = line_ransac(full_inlier, 0.03,None,get_full_length=True,transfer_indice=False,assign_radius=False)
        if free_endpt_only and len(endpt_id_list)==1:
            pair = endpt_pair(cur_line.end_points,cur_line.end_points)
            for endpt_id in endpt_id_list:
                endpt_id_to_keep = 1-endpt_id
                newLine.end_points[pair[endpt_id_to_keep]] = cur_line.end_points[endpt_id_to_keep]
        newLine.assign_radius(cur_line.radius)
        newLine.inlier = cur_line.inlier
        newLine.inlier_pred = segmented_regions
        newLine.intersection = cur_line.intersection
        log_string(LOG_FOUT,f'line {cur_Lid} is extended')
        return newLine
    else: return cur_line


######todo 1: assign edge weight W, concerning endpt distance (smaller dist - smaller weight), #inliers (more - smaller W), radius diff (smaller - smaller W, angle(90/45 degree - smaller W)

#####todo 2:form initial MST forest, iteratively increase weight ths to join trees

def Calculate_edge_weight(endpt_distance, angle, radius_diff, pred_density):
    '''endpt_distance & radius_diff: relative to pipe radius, angle in radians
    pred_density: bounded at 0.0001, 1000'''
    angle = min(abs(angle-np.pi/2),abs(angle-np.pi/4))/(np.pi/8) #[0,1]
    pred_weight = 1/max(min(pred_density,1000),0.0001)
    weight = endpt_distance*2 + angle + radius_diff + pred_weight
    return weight


def Intersect_lines_ept_search_parallel_4graph(lines, graph,scan, vertical_dist_ths=1.0, joint2end_dist_ths=1.0,min_dist2check_scan=2, scan_distribution_grid_size=0.003,log_file = None):
    #check scan clip, connect by extending lines
    '''only intersect free endpoints '''
    print("Start check line intersection in parallel")
    num_line = len(lines)
    ## get line candidate
    endpoint_list = [l.end_points for l in lines]
    endpoint_list = np.vstack(endpoint_list) #(num_lines*2,3)
    radius_list = np.vstack([l.radius for l in lines]) #(num_lines,)
    dir_list = np.vstack([l.direction for l in lines])
    endpoint1_list = endpoint_list[[i for i in range(len(endpoint_list)) if i%2]]
    pool = mp.Pool((mp.cpu_count()))
    target_dist = pool.starmap(point2lines_vertical_distance,[(dir_list,endpoint1_list,endpoint_list[i]) for i in range(num_line*2)]) #content:line id, len=num_line*2 (num_line*2,num_line)

    threshold = np.zeros((num_line*2,1))
    threshold[::2] = radius_list * max(joint2end_dist_ths, vertical_dist_ths)
    threshold[1::2] = radius_list * max(joint2end_dist_ths, vertical_dist_ths)

    target_dist = target_dist - threshold < 0 #(num_line*2,num_line)
    idx = np.arange(num_line)
    regions = [np.unique(np.concatenate((idx[target_dist[i]],idx[target_dist[i+1]]))) for i in range(0,num_line*2,2)]

    # for i in range(3,num_line):
    #     intersect_line_ind_4graph(i,lines,regions[i],scan,vertical_dist_ths,joint2end_dist_ths,scan_distribution_grid_size,log_file)

    start = timer()
    ztemp = pool.starmap(intersect_line_ind_4graph,
                         [(i,lines,regions[i],scan,vertical_dist_ths,joint2end_dist_ths,min_dist2check_scan,scan_distribution_grid_size,log_file) for i in range(num_line)])
    print(f'{timer()-start}')
    result = list(filter(None, ztemp))

    for r in result:
        for e in r:
            graph.set_edge(e[0],e[1],e[2],e[3])

    print('Finish line intersection check ~')
    return graph


def intersect_line_ind_4graph(cur_Lid,lines,region,scan,vertical_dist_ths,joint2end_dist_ths,min_dist2check_scan,scan_distribution_grid_size,log_file=None):
    region = region[region!=cur_Lid]
    if len(region) == 0:
        return
    if log_file is not None:
        LOG_FOUT= open(log_file,'a+')
    else:
        LOG_FOUT= None
    cur_line = lines[cur_Lid]

    a = cur_line.end_points[0]
    b = cur_line.end_points[1]
    dir1 = cur_line.direction

    #check angle in batch
    check_angle=np.array([lines[i].direction for i in region])
    cross = np.dot(cur_line.direction, np.transpose(np.reshape(check_angle,(-1,3))))
    angle = np.arccos(np.clip(np.abs(cross),-1.0,1.0))
    output_edge_list =[] #(frm,to,weight,pos)
    # temp_output_density = []

    log_string(LOG_FOUT,f'start processing line {cur_Lid} region {region}')
    for j,check_id in enumerate(region):
        if check_id==20 and cur_Lid == 3:
            print('')
        check_line = lines[check_id]
        longer_line,shorter_line = (cur_line,check_line) if cur_line.get_length() > check_line.get_length() else (check_line,cur_line)

        c = check_line.end_points[0]
        d = check_line.end_points[1]
        dir2 = check_line.direction

        cur_vertical_dist_ths = vertical_dist_ths
        cur_joint2end_dist_ths = joint2end_dist_ths

        nearest_pointspair = get_nearest_endpoints_pair_overlap(cur_line, check_line)
        intersect1, intersect2 = nearest_points(dir1, dir2, cur_line.end_points[nearest_pointspair[0]],check_line.end_points[nearest_pointspair[1]])
        intersect = (intersect1 + intersect2) / 2
        # np.savetxt(os.path.join('/home/user/Desktop','temp_intersect.txt'),np.concatenate((np.reshape(intersect1,(-1,3)),np.reshape(intersect2,(-1,3)),np.reshape(intersect,(-1,3))),axis=0))
        if (point2line_vertical_distance(dir1,a,intersect)<=cur_vertical_dist_ths) & (point2line_vertical_distance(dir2,c,intersect)<=cur_vertical_dist_ths):
            has_overlap = (pointline_overlap(a, b, check_line.end_points[nearest_pointspair[1]]) or pointline_overlap(c, d,cur_line.end_points[nearest_pointspair[0]]))
            joint2end_distance = max(point2point_distance(intersect,cur_line.end_points[nearest_pointspair[0]]),point2point_distance(intersect,check_line.end_points[nearest_pointspair[1]]))
            joint2end_distance2R = max(point2point_distance(intersect,cur_line.end_points[nearest_pointspair[0]])/cur_line.radius,point2point_distance(intersect,check_line.end_points[nearest_pointspair[1]])/check_line.radius)
            if not has_overlap: #check end joint condition
                if (joint2end_distance < cur_joint2end_dist_ths): #end joint
                    cur_line_full_length = point2point_distance(intersect,cur_line.end_points[1-nearest_pointspair[0]])
                    check_line_full_length = point2point_distance(intersect,check_line.end_points[1-nearest_pointspair[1]])
                    ###intersected lines length (dist of intersect - non-intersected endpt) > the other line radius
                    if check_line_full_length < cur_line.radius or cur_line_full_length < check_line.radius: # not joint
                        log_string(LOG_FOUT,'%d and %d NOT joint, one line is shorter than radius. line %d full length %.5f, line %d radius %.3f; line %d full length %.5f, line %d radius %.3f' % (cur_Lid, check_id, cur_Lid, cur_line_full_length, check_id, check_line.radius, check_id, check_line_full_length, cur_Lid, cur_line.radius))
                    else:
                        if len(check_line.intersection['end'][nearest_pointspair[1]]) >0:
                            log_string(LOG_FOUT,'%d and %d not end joint, endpoint is not free' % (cur_Lid, check_id))
                        else:
                            avg_radius = (cur_line.radius + check_line.radius)/2
                            radius_diff = abs(cur_line.radius - check_line.radius)/avg_radius
                            if joint2end_distance2R < min_dist2check_scan:
                                density = 1000
                            else:

                                scan_clip = get_pred_inbetween_endjoint_lines(scan, cur_line, check_line, nearest_pointspair,intersect,return_distribution=True,grid_size=scan_distribution_grid_size)
                                if scan_clip is not None:
                                    cur_d,check_d = scan_clip[1],scan_clip[2]
                                    density = np.average(np.concatenate((cur_d,check_d)))
                                else: density=0

                            output_edge_list.append((cur_Lid,check_id,Calculate_edge_weight(joint2end_distance2R,angle[j],radius_diff,density),tuple(intersect)+('end',)))
                            # temp_output_density.append((cur_Lid,check_id,density))
                            log_string(LOG_FOUT,f'{cur_Lid} and {check_id} end joint joint2endpoints dist {joint2end_distance:.5f} threshold {cur_joint2end_dist_ths:.5f} scan density {density:.5f}' )

                else:
                    log_string(LOG_FOUT,'%d and %d NOT end joint, nearest endpt NOT overlap & joint2endpoints dist %.5f > threshold %.5f' % (cur_Lid, check_id, joint2end_distance, cur_joint2end_dist_ths))
            else: # has overlapping, check end & mid joint condition
                crossing = line_crossing(cur_line,check_line,intersect1,intersect2)
                if crossing:
                    log_string(LOG_FOUT,f'{cur_Lid} and {check_id} cross each other, not connected')
                    continue
                ### find mid-joint line and end-joint line
                if pointline_overlap(a, b,check_line.end_points[nearest_pointspair[1]]) and \
                        not pointline_overlap(c,d,cur_line.end_points[nearest_pointspair[0]]):  # cur_line middle, check_line end
                    mid_joint_line,end_joint_line = cur_line, check_line
                    mid_joint_line_id,end_joint_line_id = cur_Lid, check_id
                    intersect2end_jointed_line_endpt_dist = abs(point2line_projection(check_line.direction,check_line.end_points[nearest_pointspair[1]],intersect))
                    intersect2mid_jointed_line_endpt_dist = abs(point2line_projection(cur_line.direction,cur_line.end_points[nearest_pointspair[0]],intersect))
                    mid_end_nearest_pointspair = nearest_pointspair
                elif pointline_overlap(c, d, cur_line.end_points[nearest_pointspair[0]]) and \
                        not pointline_overlap(a, b, check_line.end_points[nearest_pointspair[1]]):  # check_line middle cur_line end
                    mid_joint_line,end_joint_line = check_line, cur_line
                    mid_joint_line_id,end_joint_line_id = check_id, cur_Lid
                    intersect2end_jointed_line_endpt_dist = abs(point2line_projection(cur_line.direction,cur_line.end_points[nearest_pointspair[0]],intersect))
                    intersect2mid_jointed_line_endpt_dist = abs(point2line_projection(check_line.direction,check_line.end_points[nearest_pointspair[1]],intersect))
                    mid_end_nearest_pointspair = [nearest_pointspair[1],nearest_pointspair[0]]
                else:
                    intersect2cur_line_end_dist = abs(point2line_projection(cur_line.direction,cur_line.end_points[nearest_pointspair[0]],intersect))
                    intersect2check_line_end_dist =abs(point2line_projection(check_line.direction,check_line.end_points[nearest_pointspair[1]],intersect))
                    if intersect2cur_line_end_dist < intersect2check_line_end_dist:
                        mid_joint_line,end_joint_line = check_line,cur_line
                        mid_joint_line_id,end_joint_line_id = check_id, cur_Lid
                        intersect2end_jointed_line_endpt_dist = intersect2cur_line_end_dist
                        intersect2mid_jointed_line_endpt_dist = intersect2check_line_end_dist
                        mid_end_nearest_pointspair = [nearest_pointspair[1],nearest_pointspair[0]]
                    else:
                        mid_joint_line,end_joint_line = cur_line,check_line
                        mid_joint_line_id,end_joint_line_id = cur_Lid, check_id
                        intersect2end_jointed_line_endpt_dist = intersect2check_line_end_dist
                        intersect2mid_jointed_line_endpt_dist = intersect2cur_line_end_dist
                        mid_end_nearest_pointspair = nearest_pointspair

                mid_joint2end_dist_min = min(mid_joint_line.get_length() / 2, mid_joint_line.radius * 2)  # mid intersect to end-joint line endpt distance should > threshold
                if intersect2end_jointed_line_endpt_dist < cur_joint2end_dist_ths:
                    if intersect2mid_jointed_line_endpt_dist < min(cur_joint2end_dist_ths,mid_joint2end_dist_min): # end joint
                        cur_line_full_length = point2point_distance(intersect,cur_line.end_points[1-nearest_pointspair[0]])
                        check_line_full_length = point2point_distance(intersect,check_line.end_points[1-nearest_pointspair[1]])
                        ###line full length (dist of intersect - non-intersected endpt) should > than the other line radius
                        if check_line_full_length < cur_line.radius or cur_line_full_length < check_line.radius: # not joint
                            log_string(LOG_FOUT,'%d and %d NOT joint. Overlap, but one line is shorter than radius. line %d full length %.5f, line %d radius %.3f; line %d full length %.5f, line %d radius %.3f' % (cur_Lid, check_id, cur_Lid, cur_line_full_length, check_id, check_line.radius, check_id, check_line_full_length, cur_Lid, cur_line.radius))
                        else:
                            if len(check_line.intersection['end'][nearest_pointspair[1]]) >0:
                                log_string(LOG_FOUT,f'{cur_Lid} and {check_id} not end joint, endpoint is not free' )
                            else:
                                avg_radius = (cur_line.radius + check_line.radius)/2
                                radius_diff = abs(cur_line.radius - check_line.radius)/avg_radius
                                cur_dist2R = max(intersect2mid_jointed_line_endpt_dist/mid_joint_line.radius,intersect2end_jointed_line_endpt_dist/end_joint_line.radius)
                                if cur_dist2R < min_dist2check_scan:
                                    density = 1000
                                else:

                                    scan_clip = get_pred_inbetween_endjoint_lines(scan, cur_line, check_line, nearest_pointspair,intersect,return_distribution=True,grid_size=scan_distribution_grid_size)
                                    if scan_clip is not None:
                                        cur_d,check_d = scan_clip[1],scan_clip[2]
                                        density = np.average(np.concatenate((cur_d,check_d)))
                                    else: density=0

                                output_edge_list.append((cur_Lid,check_id,Calculate_edge_weight(cur_dist2R,angle[j],radius_diff,density),tuple(intersect)+('end',)))
                                # temp_output_density.append((cur_Lid,check_id,density))

                                log_string(LOG_FOUT,'%d and %d end joint. Overlap & joint2end-joint line endpt dist %.5f threshold %.5f mid_joint2end threshold %.5f, density %.5f' % (cur_Lid, check_id, intersect2end_jointed_line_endpt_dist,cur_joint2end_dist_ths,mid_joint2end_dist_min,density))


                    else: # mid joint
                        end_joint_line_full_length = point2point_distance(intersect,end_joint_line.end_points[1-mid_end_nearest_pointspair[1]])
                        if end_joint_line_full_length < mid_joint_line.radius or mid_joint_line.get_length() < end_joint_line.radius*2: #not joint
                            log_string(LOG_FOUT,'%d and %d NOT mid joint, end-joint line full length is shorter than mid-joint line radius %.5f.' % (cur_Lid,check_id,mid_joint_line.radius))
                        else: # mid joint
                            if len(end_joint_line.intersection['end'][mid_end_nearest_pointspair[1]]) >0:
                                log_string(LOG_FOUT,'%d and %d not end joint, endpoint is not free' % (cur_Lid, check_id))
                            else:
                                avg_radius = (cur_line.radius + check_line.radius)/2
                                radius_diff = abs(cur_line.radius - check_line.radius)/avg_radius
                                cur_dist2R=intersect2mid_jointed_line_endpt_dist/mid_joint_line.radius
                                if cur_dist2R < min_dist2check_scan:
                                    density = 1000
                                else:
                                    scan_clip = get_pred_inbetween_midjoint_lines(scan, end_joint_line,mid_end_nearest_pointspair[1],intersect,mid_joint_line,return_distribution=True,grid_size=scan_distribution_grid_size)
                                    if scan_clip is not None:
                                        d = scan_clip[1]
                                        density = np.average(d)
                                    else:density = 0

                                output_edge_list.append((cur_Lid,check_id,Calculate_edge_weight(cur_dist2R,angle[j],radius_diff,density),tuple(intersect)+('mid',)))
                                # temp_output_density.append((cur_Lid,check_id,density))
                                log_string(LOG_FOUT,f'{cur_Lid} and {check_id} mid joint dist to end_joint line {intersect2end_jointed_line_endpt_dist:.5f} threshold {cur_joint2end_dist_ths:.5f}, to mid_joint line ept {intersect2mid_jointed_line_endpt_dist:.5f} threshold {mid_joint2end_dist_min:.5f} density {density:.5f}')


                else: # not joint
                    log_string(LOG_FOUT,'%d and %d NOT mid joint' % (cur_Lid, check_id)+' overlap: '+str(pointline_overlap(a, b, check_line.end_points[nearest_pointspair[1]]))+str(pointline_overlap(c, d, cur_line.end_points[nearest_pointspair[0]])) +', joint to endpt dist too large %5f  %5f thres: %.5f' %(intersect2end_jointed_line_endpt_dist,intersect2mid_jointed_line_endpt_dist,cur_joint2end_dist_ths))
        else:
            log_string(LOG_FOUT,'%d and %d NOT joint, intersect to line vertical distance too large %.5f &  %.5f, threshold %.5f'%(cur_Lid,check_id,point2line_vertical_distance(dir1,a,intersect),point2line_vertical_distance(dir2,c,intersect),cur_vertical_dist_ths))
    if LOG_FOUT is not None:
        LOG_FOUT.close()
    if len(output_edge_list)>0:
        return output_edge_list
    else:return


def save_lines(lines, save_dir, save_label='', save2pkl=True,save_ind=True):
    x_total = []
    label_total = []
    radius_total = []
    for line_idx,line in enumerate(lines):
        if line.radius >0:
            num_points = int(np.linalg.norm(line.end_points[0] - line.end_points[1]) / 0.003)
            x = np.linspace(line.end_points[0], line.end_points[1], num_points)
            x_total.append(x)
            label = np.ones((num_points,1),dtype=int) * line_idx
            label_total.append(label)
            radius_total.append(np.ones((num_points,1)) * line.radius)
            if save_ind:
                save_ply(x, os.path.join(save_dir,save_label + '_%d.ply' % (line_idx)))
    x_total = np.vstack(x_total)
    label_total = np.vstack(label_total)
    radius_total = np.vstack(radius_total)
    # save_ply(x_total, os.path.join(save_dir,save_label + 'lines.ply'),label=label_total,scalar=radius_total)
    np.savetxt(os.path.join(save_dir,save_label + 'lines.txt'),np.concatenate((x_total,label_total,radius_total),-1))

    if save2pkl:
        save_data2pkl(lines, os.path.join(save_dir,save_label + '_meta.pkl'))
        # with open(save_name+'_meta.pkl', 'wb+') as f1:
        #     pickle_data = pickle.dump(lines, f1)

def save_data2pkl(data, save_name):
    '''data:(data1,data2...)'''
    if not os.path.exists(os.path.dirname(save_name)):
        os.makedirs(os.path.dirname(save_name))
    with open(save_name, 'wb+') as f1:
        pickle.dump(data, f1)

def writePipe2txt(lines,pipelist,save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    LOG_FOUT = open(os.path.join(save_dir, '../pipelist.txt'), 'w+')
    LOG_FOUT.write('idx  group  line  radius  length  endP0x  endP0y  endP0z  endP1x  endP1y  endP1z\n')
    idx = 0
    for i in range(len(pipelist)):
        # line_idx = 0
        xyz = []
        color = []
        for line_id in pipelist[i].lines:
            num_points = int(np.linalg.norm(lines[line_id].end_points[0] - lines[line_id].end_points[1]) / 0.01)
            x = np.linspace(lines[line_id].end_points[0], lines[line_id].end_points[1], num_points)
            xyz.append(x)
            color.append(np.tile(np.array([lines[line_id].radius, 0, 0]), (num_points, 1)))
            LOG_FOUT.write('%3d     %2d     %2d      %.5f     %.5f     %.5f    %.5f     %.5f    %.5f    %.5f    %.5f' % (idx, i, line_id, lines[line_id].radius, lines[line_id].get_length(), lines[line_id].end_points[0][0] , lines[line_id].end_points[0][1] , lines[line_id].end_points[0][2], lines[line_id].end_points[1][0], lines[line_id].end_points[1][1], lines[line_id].end_points[1][2]) + '\n')
            LOG_FOUT.flush()
            idx +=1
        save_ply(np.concatenate(xyz,axis=0), os.path.join(save_dir, 'pipe_%d.ply' % (i)),colors=np.concatenate(color,axis=0))
        # np.savetxt(os.path.join(save_name, 'pipe_%d.ply' % (i)))


def transform_point_cloud(xyz,transformation_matrix):
    # xyz = np.reshape(xyz,(-1,3))
    if len(xyz.shape)>1:
        new_xyz = np.ones((xyz.shape[0],xyz.shape[1]+1))
        new_xyz[:,:-1] = xyz
        return np.matmul(new_xyz,np.transpose(transformation_matrix))[:,:-1]
    else:
        new_xyz = np.ones(xyz.shape[0]+1)
        new_xyz[:-1] = xyz
        return np.matmul(new_xyz,np.transpose(transformation_matrix))[:-1]

def transform_pipe(lines,joints,transformation_matrix_list):
    '''transformation_matrix_list: list of 4x4 np.array'''
    for transformation_matrix in transformation_matrix_list:
        for l in lines:
            l.end_points[0] = transform_point_cloud(l.end_points[0],transformation_matrix)
            l.end_points[1] = transform_point_cloud(l.end_points[1],transformation_matrix)
            l.direction = transform_point_cloud(l.direction,transformation_matrix)
        for j in joints:
            if j is not None:
                j.position = transform_point_cloud(j.position,transformation_matrix)
    return lines,joints

if __name__ == '__main__':
    file_dir ='/home/user/PipeNet/results_roundDuct/point_embed_selective_sharemlp_pipenormal_radius_label_patchR_weighted_normal_euclidian_semantic/curved_duct/test/'
    file_name = 'lab_pairwise_registered_SJlab_MEP_pipe.ply_pipeClass__0.02'

