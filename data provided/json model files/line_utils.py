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
from data_augment_utils import rot_random_axis
import matplotlib.path as path
import copy
import MST_adj_matrix
# from MST_adj_matrix import Graph_subdivided

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
        self.update_length()
        return self.length

    def update_length(self):
        self.length = np.linalg.norm((self.end_points[0] - self.end_points[1]))

    def update_endpoint(self,endpt_id,pos):
        self.end_points[endpt_id] = pos
        self.update_length()
        self.update_direciton()

    def update_direciton(self):
        self.direction = (self.end_points[0] - self.end_points[1])/self.get_length()

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

def get_angle(dir1,dir2):
    '''return angle in radians, acute angle [0,pi/2]'''
    cross = np.dot(dir2, np.transpose(dir1)) #(N,)
    angle = np.arccos(np.clip(np.abs(cross), -1.0, 1.0)) #(N,)
    return angle


def get_angle_full_range(dir1,dir2):
    '''return angle in radians, angle [0,pi]'''
    cross = np.dot(dir2, np.transpose(dir1)) #(N,)
    angle = np.arccos(np.clip(cross, -1.0, 1.0)) #(N,)
    return angle

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


def point2line_projection(dir1,l1p1,l2p):
    '''projection of l2p on l1, scalar, anchor at l1p1'''
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

def project2plane(plane_p,plane_n,p):
    '''query point p projection on plane with normal plane_n pass through point plane_p'''
    d_mag = np.dot((p-plane_p),np.transpose(plane_n))
    d = d_mag*plane_n
    return p-d

def point_inside_poly(polygon_v,p,radius=0):
    '''polygon_v: (N,2)'''
    polygon_path = path.Path(polygon_v)
    return polygon_path.contains_point(p,radius=radius)

def points_inside_poly(polygon_v,p,radius=0):
    '''polygon_v: (N,2), p: (n,2)'''
    polygon_path = path.Path(polygon_v)
    return polygon_path.contains_points(p,radius=radius)



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
    np.savetxt(os.path.join(save_dir,save_label + '_lines.txt'),np.concatenate((x_total,label_total,radius_total),-1))

    if save2pkl:
        save_data2pkl(lines, os.path.join(save_dir,save_label + '_meta.pkl'))
        # with open(save_name+'_meta.pkl', 'wb+') as f1:
        #     pickle_data = pickle.dump(lines, f1)



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


def intersect_line_structure_4Graph(lines,graph, struct,pred, max_dist2wall, max_dist2ceiling,min_dist2check_scan, scan_distribution_grid_size, pool=None, log_file=None, multi_height_ceiling=False):
    print("Start check line & structure intersection in parallel")
    if log_file is not None:
        with open(log_file,'w+'):
            pass
    num_line = len(lines)

    graph.init_structure_graph(struct,multi_height_ceiling)

    # for i in range(243,244):
    #     intersect_line_structure_ind_4graph_multiCeiling(i, lines, struct,pred, max_dist2wall, max_dist2ceiling,min_dist2check_scan, scan_distribution_grid_size, log_file)

    start = timer()
    if pool is None:
        cur_pool = mp.Pool((mp.cpu_count()))
    else:
        cur_pool = pool
    if not multi_height_ceiling:
        ztemp = cur_pool.starmap(intersect_line_structure_ind_4graph,
                             [(i, lines, struct,pred, max_dist2wall, max_dist2ceiling,min_dist2check_scan, scan_distribution_grid_size, log_file) for i in range(num_line)])
    else:
        ztemp = cur_pool.starmap(intersect_line_structure_ind_4graph_multiCeiling,
                             [(i, lines, struct,pred, max_dist2wall, max_dist2ceiling,min_dist2check_scan, scan_distribution_grid_size, log_file) for i in range(num_line)])
    if pool is None:
        cur_pool.close()
        cur_pool.join()
    print(f'time {timer()-start:.3f}s')

    result = list(filter(None, ztemp))

    for r in result:
        for e in r:
            graph.set_struct_edge(e[0],e[1],e[2],e[3])

    print('Finished check line & structure intersection~')
    return graph

def intersect_line_structure_ind_4graph_multiCeiling(i,lines,struct,pred,max_dist2wall, max_dist2ceiling,min_dist2check_scan, scan_distribution_grid_size, log_file=None):
    '''check dist to 1. ceiling horizontal face; 2. wall and ceiling vertical face'''
    if log_file is not None:
        LOG_FOUT= open(log_file,'a+')
    else:
        LOG_FOUT= None
    output_edge_list =[] #(frm line name,to struct name,weight,pos)

    cur_line = lines[i]
    a,b = cur_line.end_points
    dir = cur_line.direction
    r = cur_line.radius

    is_vertical = False
    if np.rad2deg(get_angle(dir,[0,0,1])) < 10:
        is_vertical = True

    ceiling_height = struct['Wall'][0]['height']
    ceiling_base_thickness = struct['Ceiling'][0]['thickness']

    ceiling_profiles = []
    for c in struct['MultiHeightCeiling']:
        ceiling_profiles.append(np.vstack([[obj['x'],obj['y']] for obj in c['baseProfile']]))

    ceilings_thickness = np.array([obj['thickness'] for obj in struct['MultiHeightCeiling']])

    max_dist2wall *= r
    max_dist2ceiling *= r

    ########## distance to ceiling horizontal face
    dist2ceiling = sys.maxsize
    nearest_ceiling_id = None
    nearest_endpt2ceiling = None
    if is_vertical:
        pos = 'end'
    else:
        pos = 'makeup'
    for ceiling_id,ceiling in enumerate(struct['MultiHeightCeiling']):
        cur_ceiling_profile =ceiling_profiles[ceiling_id]
        cur_ceiling_height = ceiling_height + ceiling_base_thickness - ceiling['thickness']
        cur_ceiling_thickness = ceiling['thickness']

        # if endpoint within ceiling profile, find dist to horizontal face
        if point_inside_poly(cur_ceiling_profile,a[:-1]) or point_inside_poly(cur_ceiling_profile,b[:-1]):
            for x in range(2):
                if abs(ceiling_height-cur_line.end_points[x][-1]) < dist2ceiling and point_inside_poly(cur_ceiling_profile,cur_line.end_points[x][:-1]) and cur_line.end_points[x][-1] < cur_ceiling_height:
                    dist2ceiling = ceiling_height-cur_line.end_points[x][-1]
                    nearest_endpt2ceiling = x
                    nearest_ceiling_id = ceiling_id

    if dist2ceiling < max_dist2ceiling:
        if pos=='end':
            t = (ceiling_height - a[-1])/dir[-1] # line equatio: P = a + dir*t
            intersect = a + dir*t
            if not point_inside_poly(ceiling_profiles[nearest_ceiling_id],intersect[:-1]): # line extension not within ceiling profile
                pos = 'makeup'
        if pos=='makeup':
            plane_p = np.array([struct['Ceiling'][0]['baseProfile'][0]['x'],struct['Ceiling'][0]['baseProfile'][0]['y'],ceiling_height])
            intersect = point2plane_projection(plane_p,np.array([0,0,1]),cur_line.end_points[nearest_endpt2ceiling])

        dist2ceiling2R = dist2ceiling/r
        if dist2ceiling2R > min_dist2check_scan:
            scan_clip = get_pred_inbetween_midjoint_lines(pred,cur_line,nearest_endpt2ceiling,intersect,return_distribution=True,grid_size = scan_distribution_grid_size)
            if scan_clip is not None:
                d = scan_clip[1]
                density = np.average(d)
            else:
                density = 0
        else: density = 1000

        cost  = Calculate_edge_weight(dist2ceiling2R,1,1,density,component_weight=[2,1,1])

        if pos == 'makeup':
            intersect = tuple(lines[i].end_points[nearest_endpt2ceiling]) + tuple(intersect)

        output_edge_list.append(('%d-%d'%(i,nearest_endpt2ceiling),'ceiling-%d'%nearest_ceiling_id,cost,tuple(intersect)+(pos,)))
        log_string(LOG_FOUT,f'line {i} and ceiling {nearest_ceiling_id} horizontal face intersect, pos {pos}, dist {dist2ceiling:.5f}={dist2ceiling2R:.3f}R < ths {max_dist2ceiling:.3f}, cost {cost:.3f}')
    else:
        if nearest_endpt2ceiling is not None:
            log_string(LOG_FOUT, f'\tline {i} and ceiling {nearest_ceiling_id} horizontal face NOT intersect, pos {pos}, dist {dist2ceiling:.5f} > ths {max_dist2ceiling:.5f}')
        else:log_string(LOG_FOUT, f'\tline {i} is OUT of ceiling')

    ## distance to wall and ceiling vertical face
    wall_lines = []
    walls_thickness = []
    wall_base_height = []
    wall_lines2obj_id_map = []
    for wallid, wall in enumerate(struct['Wall']):
        wall_lines.append([[wall['startPoint']['x'],wall['startPoint']['y']],[wall['endPoint']['x'],wall['endPoint']['y']]]) #list (2,2)
        walls_thickness.append(wall['width'])
        wall_base_height.append(0)
        wall_lines2obj_id_map.append(wallid)
    num_wall = len(walls_thickness)

    for ceiling_id,ceiling in enumerate(struct['MultiHeightCeiling']):
        for vid,v in enumerate(ceiling_profiles[ceiling_id]):
            walls_thickness.append(0)
            wall_base_height.append(ceiling_height + ceiling_base_thickness - ceiling['thickness'])
            wall_lines2obj_id_map.append(ceiling_id)
            try: wall_lines.append([v,ceiling_profiles[ceiling_id][vid+1]])
            except: wall_lines.append([v,ceiling_profiles[ceiling_id][0]])
    num_ceiling = len(walls_thickness) - num_wall

    for x in range(2):
        dist2wall = sys.maxsize
        nearest_wall_id = None
        nearest_endpt2wall = None

        x_point = cur_line.end_points[x]
        x2line_dist = [point2line_distance_2D(x_point[:-1],np.array(line[0]),np.array(line[1])) for line in wall_lines]

        for id,dist in enumerate(x2line_dist):
            if dist > dist2wall:
                continue

            line = wall_lines[id]
            ## check wall line or ceiling line
            if id < num_wall:
                # wall line. check if endpoint overlap with wall line
                if pointline_overlap(np.array([line[0][0],line[0][1],0]),np.array([line[1][0],line[1][1],0]),np.array([x_point[0],x_point[1],0])):
                    nearest_wall_id = id
                    nearest_endpt2wall = x
                    dist2wall = abs(dist - walls_thickness[id]/2)
            else:
                # ceiling line. check if endpoint within vertical face profile
                try:
                    if pointline_overlap(np.array([line[0][0],line[0][1],0]),np.array([line[1][0],line[1][1],0]),np.array([x_point[0],x_point[1],0])) and x_point[-1] > wall_base_height[id]:
                        nearest_wall_id = id
                        nearest_endpt2wall = x
                        dist2wall = abs(dist - walls_thickness[id]/2)
                except:
                    print(line,x_point)

        if nearest_wall_id < num_wall:
            tar = 'wall'
        else:
            tar = 'ceiling'

        if dist2wall < max_dist2wall:
            line = wall_lines[nearest_wall_id]
            base_height = wall_base_height[nearest_wall_id]
            startpoint = np.array([line[0][0],line[0][1],0])
            endpoint = np.array([line[1][0],line[1][1],0])
            wall_n = get_wall_normal(startpoint,endpoint)
            if is_vertical:
                pos = 'makeup'
            else:
                wall_dir = get_line_direction(startpoint, endpoint)
                dir_proj = dir - np.array([0,0,dir[-1]]) #line dir projection on xy plane: minus z-axis component
                if np.rad2deg(get_angle(dir_proj,wall_dir)) < 45:
                    # line near parallel with wall
                    pos = 'makeup'
                else: pos = 'end'

            intersect = line2plane_intersection(startpoint, wall_n, a, dir)
            if pos == 'end' and (not pointline_overlap(startpoint,endpoint,intersect) or intersect[-1] < base_height or intersect[-1] > ceiling_height): # line extension not within face profile
                pos = 'makeup'

            if pos=='makeup': # intersect is projection of nearest endpt on wall
                intersect = point2plane_projection(startpoint, wall_n, cur_line.end_points[nearest_endpt2wall])

            dist2wall2R = dist2wall/r

            if dist2wall2R > min_dist2check_scan:
                scan_clip = get_pred_inbetween_midjoint_lines(pred,cur_line,nearest_endpt2wall,intersect,grid_size=scan_distribution_grid_size,return_distribution=True)
                if scan_clip is not None:
                    d = scan_clip[1]
                    density = np.average(d)
                else:
                    density = 0
            else: density = 1000

            cost = Calculate_edge_weight(dist2wall2R,1,1,density,component_weight=[2,1,1])

            if pos=='makeup':
                intersect = tuple(lines[i].end_points[nearest_endpt2wall]) + tuple(intersect)

            output_edge_list.append(('%d-%d'%(i,nearest_endpt2wall),'%s-%d'%(tar,wall_lines2obj_id_map[nearest_wall_id]),cost,tuple(intersect)+(pos,)))

            log_string(LOG_FOUT,f'line {i}-endpt {nearest_endpt2wall} and vertical {tar} {wall_lines2obj_id_map[nearest_wall_id]} intersect, pos {pos}, dist {dist2wall:.5f}={dist2wall2R:.3f}R ths {max_dist2wall:.5f}, cost {cost:.3f}')
        else:
            log_string(LOG_FOUT,f'\tline {i}-endpt {nearest_endpt2wall} and vertical {tar} {wall_lines2obj_id_map[nearest_wall_id]} NOT intersect, dist {dist2wall:.5f}={dist2wall/r:.3f}R > ths {max_dist2wall:.5f}')

    return output_edge_list


def get_wall_normal(startpoint,endpoint):
    '''startpoint&endpoint: on xy plane
     n*wall_dir = 0 == nx*dx + ny*dy = 0'''
    wall_dir = get_line_direction(startpoint, endpoint)
    if abs(wall_dir[0]) < 1e-3:
        nx = 1
        ny = (-wall_dir[0]*nx)/wall_dir[1]
    elif abs(wall_dir[1]) < 1e-3:
        ny = 1
        nx = (-wall_dir[1]*ny)/wall_dir[0]
    return np.array([nx,ny,0])/np.linalg.norm(np.array([nx,ny,0]))

def get_line_direction(startpoint, endpoint):
    return (startpoint-endpoint)/np.linalg.norm(startpoint-endpoint)

def point2plane_projection(plane_point, plane_n, point):
    return point - np.dot(plane_n,(point-plane_point))*plane_n

def line2plane_intersection(plane_point, plane_n, line_point, line_dir):
    t = np.dot((plane_point - line_point),plane_n)/np.dot(line_dir, plane_n)
    return line_point + line_dir*t

def point2plane_distance(plane_point, plane_n, point):
    return np.linalg.norm(np.dot(plane_n,(point-plane_point))*plane_n)

def point2line_distance_2D(query_p, line_p1, line_p2, line_dir=None):
    if line_dir is None:
        line_dir = get_line_direction(line_p1, line_p2)
    cross = np.cross(line_dir, (line_p1 - query_p))
    return np.linalg.norm(cross)





if __name__ == '__main__':


    print('FIN')
