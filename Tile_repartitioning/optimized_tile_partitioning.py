import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
from scipy.spatial import distance
import collections

error_frame = False


# test codes for MNC parti
def try_nx_for_getting_results(horizontal_chords,
                               vertical_chords,
                               concave_vertices,
                               convex_vertics,
                               colinear_vertices,
                               boundary,
                               neighbour_list,
                               is_holes_contains,
                               holes,
                               video_name,
                               frame_num,
                               threshold,
                               is_fov,
                               is_buffer):
    G = nx.Graph()

    max_independent_set = []

    for i, h_chord in enumerate(horizontal_chords):

        intersect_list = []
        intersect_list.append(h_chord)
        h_r = h_chord[0][0]
        h_c_min = np.min(np.asarray([h_chord[0][1], h_chord[1][1]]))
        h_c_max = np.max(np.asarray([h_chord[0][1], h_chord[1][1]]))

        G.add_node(i, bipartite=1)

        # count = 0
        for j, v_chord in enumerate(vertical_chords):

            v_c = v_chord[0][1]
            v_r_min = np.min(np.asarray([v_chord[0][0], v_chord[1][0]]))
            v_r_max = np.max(np.asarray([v_chord[0][0], v_chord[1][0]]))
            G.add_node(j + len(horizontal_chords), bipartite=0)
            if ((v_c <= h_c_max) and (v_c >= h_c_min)) and ((h_r >= v_r_min) and (h_r <= v_r_max)):
                intersect_list.append(v_chord)
                # count+=1
                # added_v = j

                G.add_edge(i, j + len(horizontal_chords))
        # if count==1:
        #     G.remove_edge(i,added_v)

    if len(horizontal_chords) == 0:
        for j, v_chord in enumerate(vertical_chords):
            v_c = v_chord[0][1]
            v_r_min = np.min(np.asarray([v_chord[0][0], v_chord[1][0]]))
            v_r_max = np.max(np.asarray([v_chord[0][0], v_chord[1][0]]))
            G.add_node(j + len(horizontal_chords), bipartite=0)

    # nx.draw(G, with_labels=True, font_weight='bold')
    # plt.show()

    l = nx.connected_components(G)
    # print(nx.is_connected(G))
    # print(nx.number_connected_components(G))
    # print(nx.connected_components(G))
    # print(nx.node_connected_component(G,0))
    S = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    # extract only the maximum subgraph
    max_nodes = 0

    # if len(S)>0:
    #     for s in S:
    #         if max_nodes<len(s):
    #             max_nodes = len(s)
    #             new_s = s
    # S = [new_s]

    debug = True
    all_independet_chord = []
    all_unmatched_concave_chord = []

    # form the for loop, try to find all the max independan chords for a given rectilinear polygon
    for s in S:

        if nx.is_connected(s):

            # find the maximum matching
            M = nx.Graph()
            maximum_matching = nx.bipartite.maximum_matching(s)
            maximum_matching_list = []

            for i, j in maximum_matching.items():
                maximum_matching_list += [(i, j)]
            M.add_edges_from(maximum_matching_list)
            maximum_matching = M.edges()

            # breaking up into two sets
            H, V = bipartite.sets(s)

            free_vertices = []
            for u in H:
                temp = []
                for v in V:
                    if (u, v) in maximum_matching or (v, u) in maximum_matching:
                        temp += [v]
                if len(temp) == 0:
                    free_vertices += [u]
            for u in V:
                temp = []
                for v in H:
                    if (u, v) in maximum_matching or (v, u) in maximum_matching:
                        temp += [v]
                if len(temp) == 0:
                    free_vertices += [u]

            # finding the maximum independent set
            max_independent = []
            maximum_matching = list(maximum_matching)
            while len(free_vertices) != 0 or len(maximum_matching) != 0:
                if len(free_vertices) != 0:
                    u = free_vertices.pop()
                    max_independent += [u]
                else:
                    u, v = maximum_matching.pop()
                    s.remove_edge(u, v)
                    max_independent += [u]
                x = [n for n in s.neighbors(u)]
                for v in x:  # s.neighbors(u):
                    s.remove_edge(u, v)
                    for h in s.nodes():
                        if (v, h) in maximum_matching:
                            maximum_matching.remove((v, h))
                            free_vertices += [h]
                        if (h, v) in maximum_matching:
                            maximum_matching.remove((h, v))
                            free_vertices += [h]

            independent_chords = []

            # check if the received max indpendent coordinates are horizontal. if not t
            for i in max_independent:
                if (i >= len(horizontal_chords)):
                    independent_chords += [vertical_chords[i - len(horizontal_chords)]]
                else:
                    independent_chords += [horizontal_chords[i]]
            # if len(s) == 2:
            #     print(1)
            all_independet_chord.extend(independent_chords)

    # find all the unmatched concave vertices in the rectlinear polygon
    # also, add those cocave vertice to draw the independant chords to the colinear vertices set
    # if there are holes presented, assume those vertices are also concave vertices. and include them in teh exsisting concave vector
    if is_holes_contains:
        for hole in holes:
            for vertex in hole:
                concave_vertices.append(vertex)

    unmatched_concave_vertices = [i for i in concave_vertices]

    for i, j in all_independet_chord:
        if i in unmatched_concave_vertices:
            unmatched_concave_vertices.remove(i)
            colinear_vertices.append(i)
        if j in unmatched_concave_vertices:
            unmatched_concave_vertices.remove(j)
            colinear_vertices.append(j)

    # non of the nearest chord that we draw should not pass through the hole area. In that case,
    # temporarally we add horizontal edges of the holes to the all_independent chord set, so that, new vertical
    # lines defined now will not pass through the holes.
    # later before returning the all independent chords, all the newly added chords should be removed from the list
    # also, since have already included all the unmatched vertices of holes into the concave vertices vector,
    # we need to add all the coordinate pairs of the holes to the neighbourlist in order to avoid selecting the vertex of the
    # same hole as the nearest colinear vertex
    if is_holes_contains:
        for hole in holes:
            all_independet_chord.append([hole[0], hole[1]])
            all_independet_chord.append([hole[3], hole[2]])

            # Adding vertices of all hole to the neighbour_list
            neighbour_list.append(hole[0])
            neighbour_list.append(hole[1])
            neighbour_list.append(hole[2])
            neighbour_list.append(hole[3])

    # for the remainnig unmatched concave vertices, find the nearset orthogonal independent chord or
    # colinear vertex to draw all the remaining possible vertical chords

    if debug:
        nearest_chord = []
        for i in unmatched_concave_vertices:
            # if i == [11, 6]:
            #     print(1)
            dist = 0
            nearest_distance = math.inf
            # select all the independent chords that have been already found
            for j in all_independet_chord:
                if j[0][0] == j[1][0]:  # check whether are we relating to a horizontal chord
                    temp1, temp2 = j[0], j[1]

                    t = abs(np.asarray(temp1) - np.asarray(i))

                    # if abs(i[0] - temp1[0]) < nearest_distance and (
                    #         (i[1] <= temp1[1] and i[1] >= temp2[1]) or (i[1] >= temp1[1] and i[1] <= temp2[1])) \
                    #         and distance.euclidean(temp1, i) != 1 and distance.euclidean(temp2, i) != 1:
                    if (abs(i[0] - temp1[0]) < nearest_distance) and (abs(i[0] - temp1[0]) > 0) and (
                            (i[1] < temp1[1] and i[1] > temp2[1]) or (i[1] > temp1[1] and i[1] < temp2[1])):

                        middle_found = False
                        # check wether there are any intermediate chords between the selected unconected coord and the independent chord
                        for g in all_independet_chord:

                            if (g[0][0] == j[0][0]) and (g[0][1] == j[0][1]) and (g[1][0] == j[1][0]) and (
                                    g[1][1] == j[1][1]):
                                continue
                            else:
                                # extract only the horizontal coordinates
                                if g[0][0] == g[1][0]:

                                    g1, g2 = g[0], g[1]

                                    # check if there is a middle chord between the concave point and the independent chord considers above
                                    # ideally, the rod between chord j and concave point should not intersect any of the chord g
                                    # first main () in the following condition consider the vertical location (x location accorfding to a matrix) and
                                    # second main () in the following condition considers the horizontal location
                                    if ((g1[0] > i[0] and g1[0] < temp1[0]) or (
                                            g1[0] < i[0] and g1[0] > temp1[0])) and (
                                            (g1[1] > i[1] and g2[0] < i[1]) or (g1[1] < i[1] and g2[1] > i[1])):
                                        middle_found = True
                                        break

                        if not middle_found:
                            # check whether there are any intermediate chord beteween the rod and the concave point. If there
                            # any colinear/concave there is a middle

                            middle_coord_found = False

                            check_list = []
                            check_list.extend(colinear_vertices)
                            check_list.extend(convex_vertics)
                            check_list.extend(concave_vertices)

                            for h in check_list:
                                if h[0] == i[0] or h[1] != i[1]:
                                    continue
                                else:
                                    if (h[0] > i[0] and h[0] < temp1[0]) or (h[0] < i[0] and h[0] > temp1[0]):
                                        middle_coord_found = True
                                        break
                            if not middle_coord_found:
                                nearest_distance = abs(i[0] - temp1[0])
                                dist = temp1[0] - i[0]
                                break

                        # for u in range(len(x)):
                        #     if x[i] == x[u] and (
                        #             y[i] < y[u] and y[u] < y[temp1] or y[temp1] < y[u] and y[u] < y[i]):
                        #         middles.append(u)
                        # if len(middles) == 0:
                        #     nearest_distance = abs(y[i] - y[temp1])
                        #     dist = y[temp1] - y[i]

                        # if nearest_distance_temp < nearest_distance:
                        #     nearest_distance = nearest_distance_temp

            if nearest_distance != math.inf:
                nearest_chord.append((i, dist))
            else:
                # if i == [11, 6]:
                #     print(1)
                # try to find if there are any colinear vertices matching the unconnected concave vertices.
                for k in colinear_vertices:
                    middle_found = False
                    if k[1] == i[1] and k[0] - i[0] != 0 and abs(
                            k[0] - i[0]) < nearest_distance and check_neighbourhood(k, i, neighbour_list, video_name,
                                                                                    frame_num, threshold,is_fov,is_buffer):
                        if abs(k[0] - i[0]) == 1:
                            nearest_distance = abs(k[0] - i[0])
                            dist = (k[0] - i[0])
                            break
                        else:
                            check_list = []
                            check_list.extend(colinear_vertices)
                            check_list.extend(convex_vertics)
                            check_list.extend(concave_vertices)
                            # if i == [7, 5]:
                            #     print(1)
                            for h in check_list:
                                if h[0] == k[0] or h[1] != k[1]:
                                    continue
                                else:
                                    if (h[0] > i[0] and h[0] < k[0]) or (h[0] < i[0] and h[0] > k[0]):
                                        middle_found = True
                                        break

                            if not middle_found:
                                # if nearest_distance > abs(k[0] - i[0]):
                                nearest_distance = abs(k[0] - i[0])
                                dist = (k[0] - i[0])
                                break

                if nearest_distance != math.inf:
                    nearest_chord.append((i, dist))

        print("The minimum partitioned rectillinear polygon")

        inpdendnet_hori_chords = []
        # for i in max_independent:
        #     if i < len(horizontal_chords):
        #         independent_chords.append(horizontal_chords[i])

        # convert nearset chords to cartisiean coordinate format
        nearest_chord_cartisian_chord = []
        for i in nearest_chord:
            x = [i[0][1], i[0][1]]
            y = [(i[0][0]), ((i[0][0] + i[1]))]
            nearest_chord_cartisian_chord.append([[y[0], x[0]], [y[1], x[1]]])

        if is_holes_contains:
            for hole in holes:
                all_independet_chord.remove([hole[0], hole[1]])
                all_independet_chord.remove([hole[3], hole[2]])

        # plot the graph given the vertices and chord
        # plot_the_graph(convex=convex_vertics,
        #                concave=concave_vertices,
        #                colinear=colinear_vertices,
        #                indpendent_set=all_independet_chord,
        #                nearest_set=nearest_chord,
        #                boundary=boundary)

    global error_frame
    error_frame = False

    return all_independet_chord, nearest_chord_cartisian_chord


# check whether the given coordinates are in the neighbourhoods of the given boundary line
# if the checked vertices are in the neighbourhood and on the boundary line, we omit that coordinate.
def check_neighbourhood(k, i, neighbour_list, video_name, frame_num, threshold,is_fov,is_buffer):
    if k in neighbour_list and i in neighbour_list:
        ind_k = neighbour_list.index(k)
        ind_i = neighbour_list.index(i)
    else:
        global error_frame
        # log the error frame info

        if is_fov:
            log_path = '/Users/ckat9988/Documents/Research/FoV/Inter_Process_Data/Viewport_maps/Partitionined_frame_data/' \
                       'FOV_partitioning/' \
                       'train/closed/Adpative_threshold_partitioning_2/1_Partitioned_frames'
        elif is_buffer:
            log_path = '/Users/ckat9988/Documents/Research/FoV/Inter_Process_Data/Viewport_maps/Partitionined_frame_data/' \
                       'Buffer_partitioning/' \
                       'train/closed/Adpative_threshold_partitioning/1_Partitioned_frames'
        else:
            log_path = '/Users/ckat9988/Documents/Research/FoV/Inter_Process_Data/Viewport_maps/Partitionined_frame_data/' \
                       'OOV_partitioning/' \
                       'train/closed/Adpative_threshold_partitioning/1_Partitioned_frames'

        f = open(log_path + '/' + 'error_frames.txt', "a+")

        string = 'video: ' + video_name + ' frame: ' + str(frame_num) + ' threshold: ' + str(threshold) + '\n'
        if not error_frame:
            f.write(string)
            f.flush()
            error_frame = True
            f.close()

        ind_k = 0
        ind_i = 0

    if abs(ind_k - ind_i) == 1:
        return False
    else:
        return True


# given the convex and concve vertices this algorithm tries to find all the colinear vertice
def plot_the_graph(convex, concave, colinear, indpendent_set, nearest_set, boundary):
    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    ax.set_ylim(21, 0)
    ax.set_xlim(0, 11)
    x = []
    y = []
    for i in (convex):
        x.append(i[1])
        y.append(i[0])
        plt.scatter(x, y, color='r')
    x = []
    y = []
    for j in (concave):
        x.append(j[1])
        y.append(j[0])
        plt.scatter(x, y, color='g')
    x = []
    y = []
    for k in (colinear):
        x.append(k[1])
        y.append(k[0])
        plt.scatter(x, y, color='b')

    for i in indpendent_set:
        x = [i[0][1], i[1][1]]
        y = [(i[0][0]), (i[1][0])]

        plt.plot(x, y, color='black', linewidth=3, marker='o')

    for i in nearest_set:
        x = [i[0][1], i[0][1]]

        if i[1] < 0:
            y = [(i[0][0]), ((i[0][0] + i[1]))]
        else:
            y = [(i[0][0]), ((i[0][0])) + i[1]]
        plt.plot(x, y, color='black', linewidth=3, marker='o')

        # plt.grid()

    ax.tick_params(axis='both',
                   labelsize=40)
    plt.imshow(boundary)
    plt.show()

    return

# def main():
#     try_nx_for_getting_results()
#     return
#
#
# if __name__ == main():
#     main()

# taking max matching set
# creating the bipartite graph
# horizontal_chords = [[[3, 4], [3, 7]], [[4, 3], [4, 7]], [[5, 3], [5, 7]],
#                      [[8, 4], [8, 7]], [[10, 4], [10, 7]],[[11, 4], [11, 7]]]
#
# vertical_chords = [[[3, 4], [8, 4]], [[4, 3], [5, 3]], [[4, 7], [5, 7]], [[9, 4], [10, 4]],[[8,7],[10,7]]]
#
# concave_vertices = [[3, 4], [2, 5], [3, 7], [4, 7], [5, 7], [8, 7], [9, 4], [8, 4], [7, 3],
#                     [5, 3], [4, 3], [10, 7], [10, 4],[11,7],[11,4]]
# convex_vertics = [[1, 4], [1, 5], [2, 8], [3, 8], [4, 8], [5, 8], [8, 8], [10, 8], [10, 2], [9, 2], [8, 1], [7, 1],
#                   [5, 2], [4, 2], [3, 3],[11,3],[12,3],[11,8],[12,8]]
# colinear_vertices = [[2, 6], [2, 7], [6, 7], [7, 7], [9, 8], [10, 7], [10, 4], [10, 3], [9, 3],
#                      [8, 3], [8, 2], [7, 2], [6, 3], [2, 4],[12,4],[12,5],[12,6],[12,7]]
