import numpy as np
import matplotlib.pyplot as plt

# direciton definitions
right = 1
down = 2
left = 3
up = 4


# Supportive funcitons ==========#
# check the given coordinate is in the given list. if not append
def check_and_append_coord_in_list(c1, list):
    if not c1 in list:
        list.append(c1)
    return list


# Starting place for the recursive function
def find_coord(neighbour_list, concave_coord, convex_coord, colinear_coord, independent_chord, nearest_chord,
               is_holes_contains, holes):
    all_vertices = []
    all_rectangles = []
    # put all the coordinates to one list:boundary coordinates
    for b_ind, b in enumerate(neighbour_list):
        all_vertices.append(b[0])
    # put all the coordinates to one list:independent coordinates
    for c1, c2 in independent_chord:
        all_vertices = check_and_append_coord_in_list(c1, all_vertices)
        all_vertices = check_and_append_coord_in_list(c2, all_vertices)
    # put all the coordinates to one list:neares coordinates
    for c1, c2 in nearest_chord:
        all_vertices = check_and_append_coord_in_list(c1, all_vertices)
        all_vertices = check_and_append_coord_in_list(c2, all_vertices)

    # add the hole vertices to the neighbour list if holes are presented and also consider them as the concave vertices
    left_upper_coord_of_holes = []
    if is_holes_contains:
        for hole in holes:
            # including data to the neighbout list
            neighbour_list.append([hole[0], hole[1]])
            neighbour_list.append([hole[1], hole[2]])
            neighbour_list.append([hole[2], hole[3]])
            neighbour_list.append([hole[3], hole[0]])

            # including hole coordinate to the concave coordinate list
            concave_coord.append(hole[0])
            concave_coord.append(hole[1])
            concave_coord.append(hole[2])
            concave_coord.append(hole[3])

            left_upper_coord_of_holes.append(hole[0])

    for b_coord_ind, b_coord in enumerate(all_vertices):
        # if b_coord != [12,3]:
        #     continue

        # if the holes are presented remove the left upper corner point of all the holes
        if b_coord in left_upper_coord_of_holes:
            continue

        # find any (nearest coordinates)between the above coordinate and the next neihbour
        # should select those coordinates before proceed to finding coordinates
        left_upper = [b_coord]
        # we know all the nearest chords are vertical. therefore, if and only if above b_cord and its neighbour
        # coordinate is hoizontal try to find the coordinate.
        # find the neareset boundary coordinate near to the same coordinate considered above which is at the same vertical (x) coordinate level.
        closest_coord = [100, 100]
        for b_coord_near_ind, b_coord_near in enumerate(neighbour_list):
            # and (b_coord_near[1][0] == b_coord[1][0])
            if (b_coord_near[0][0] == b_coord[0]) and (b_coord_near[0][1] > b_coord[1]):

                if closest_coord[0] > b_coord_near[0][0]:
                    closest_coord = b_coord_near[0]
        # find if there are any nearest chord coordinate which lies between the left upper coordinate and its, nearest boundary coordinate
        # at the same vertical level.
        valid_n_cords = []
        for n_cord_ind, n_cord in enumerate(nearest_chord):
            valid_n_cord_found = False
            if (n_cord[0][0] == b_coord[0]) and (n_cord[0][1] > b_coord[1]) and (n_cord[0][1] < closest_coord[1]):
                valid_n_cord = n_cord[0]
                valid_n_cord_found = True
            elif (n_cord[1][0] == b_coord[0]) and (n_cord[1][1] > b_coord[1]) and (
                    n_cord[1][1] < closest_coord[1]):
                valid_n_cord = n_cord[1]
                valid_n_cord_found = True

            # if only we find any valid nearese coordinate cordinates we appeand it to the valid_n_cord
            if valid_n_cord_found:
                valid_n_cords.append(valid_n_cord)

        # following coordinates should be taken as the left upper coordinates to track the rectangles
        coords_to_track = []
        # append the first coordinate from the boundary line
        coords_to_track.append(b_coord)
        # append other extra nearest chord coordinations which lie between the considered boundary coordinate and the
        # closet point for that.
        if len(valid_n_cords) > 0:
            coords_to_track.append(valid_n_cord)

        # start recursive function to find the valid coordinates
        for valid_coord_id, valid_coord in enumerate(coords_to_track):
            rectangle_cord = []
            rectangle_cord.append(valid_coord)

            # if valid_coord_id == 0:
            #     continue

            hunt_rectnalge_coord(direction=right,
                                 current_cord=valid_coord,
                                 left_upper=valid_coord,
                                 neighbour_list=neighbour_list,
                                 rectangle_cord_list=rectangle_cord,
                                 all_vertices=all_vertices,
                                 convex_cord=convex_coord,
                                 concave_cord=concave_coord,
                                 independent_chord=independent_chord,
                                 nearest_chord=nearest_chord)

            if len(rectangle_cord) == 4:
                all_rectangles.append(rectangle_cord)
                print(rectangle_cord)

        # print(1)

    return all_rectangles


# direciton definitions
# right = 1
# down = 2
# left = 3
# up = 4

# recursive function to find the rectangular areas derived.
def hunt_rectnalge_coord(direction, current_cord, left_upper, neighbour_list, rectangle_cord_list, all_vertices,
                         convex_cord,
                         concave_cord, independent_chord,
                         nearest_chord):
    # if it is not the direction of up, we need to find more coordinates

    candidate_coordinates = find_cordinates_based_on_direction(direction=direction,
                                                               current_cord=current_cord,
                                                               all_vertices=all_vertices)
    prev_cord = current_cord
    if len(candidate_coordinates) > 0:
        for candidate_coordinates_id, candidate_coordinate in enumerate(candidate_coordinates):

            if candidate_coordinates_id == 0:
                last_cord = prev_cord

            # if it detectes convex coordinates which can be connected outside the class those coordinates should be avoided
            # 1. check whether the candidate coordinate in convex
            # 2. check whether the prev coordinate is convex
            # 3. check whether the 2 codinates are neighbour coordinates or not
            # 4. check whther the candiate coordinate is the left upper
            # if prev_cord == [13, 4]:
            #     print(1)
            #
            #     print(candidate_coordinate in convex_cord)
            #     print(prev_cord in convex_cord)
            #     print(not ([prev_cord, candidate_coordinate] in neighbour_list))
            #     print(not candidate_coordinate == left_upper)

            if (candidate_coordinate in convex_cord) and (prev_cord in convex_cord) and (
                    not ([last_cord,
                          candidate_coordinate] in neighbour_list)):# and candidate_coordinates_id == 0:  # and (not candidate_coordinate == left_upper):

                if (not (check_for_convex_coord_out_of_polygon(prev_cord,
                                                          candidate_coordinate,
                                                          candidate_coordinates,
                                                          all_vertices,
                                                          independent_chord, nearest_chord
                                                          ))):
                    return False

            # check if the prev coord and candidate coordinate is crossing any of the coordinate
            # if it crosses any coordinate remove ths chord. Also, check whether if it pass through the coliner vertices boundary
            if cross_chords(prev_cord, candidate_coordinate, independent_chord, nearest_chord, neighbour_list):
                return False

            if direction < 4:
                # finding the coordinate at the same level.
                # based on the direction we need to find the coordinations changes

                # check if the detected coordinates are concave coordinates and are they connected internally with a chord
                if ((candidate_coordinate in concave_cord) and (prev_cord in concave_cord)) or (
                        [candidate_coordinate, prev_cord] in nearest_chord) or (
                        [prev_cord, candidate_coordinate] in nearest_chord):
                    # (candidate_coordinate in nearest_chord) and (prev_cord in concave_cord)) or (
                    # (prev_cord in nearest_chord) and (candidate_coordinate in concave_cord)):
                    if ([candidate_coordinate, prev_cord] in independent_chord) or (
                            [prev_cord, candidate_coordinate] in independent_chord) or (
                            [candidate_coordinate, prev_cord] in nearest_chord) or (
                            [prev_cord, candidate_coordinate] in nearest_chord) or (
                            [prev_cord, candidate_coordinate] in neighbour_list) or (
                            [candidate_coordinate, prev_cord] in neighbour_list):
                        rectangle_cord_list.append(candidate_coordinate)
                        prev_cord = candidate_coordinate
                        # run the next coordination
                        coordinate_statues = hunt_rectnalge_coord(direction=direction + 1,
                                                                  current_cord=candidate_coordinate,
                                                                  left_upper=left_upper,
                                                                  neighbour_list=neighbour_list,
                                                                  rectangle_cord_list=rectangle_cord_list,
                                                                  all_vertices=all_vertices,
                                                                  convex_cord=convex_cord,
                                                                  concave_cord=concave_cord,
                                                                  independent_chord=independent_chord,
                                                                  nearest_chord=nearest_chord)
                        if not coordinate_statues:
                            rectangle_cord_list.pop()

                            continue
                        else:
                            return True
                    else:
                        return False

                else:
                    prev_cord = candidate_coordinate
                    rectangle_cord_list.append(candidate_coordinate)
                    # run the next coordination
                    coordinate_statues = hunt_rectnalge_coord(direction=direction + 1,
                                                              current_cord=candidate_coordinate,
                                                              left_upper=left_upper,
                                                              neighbour_list=neighbour_list,
                                                              rectangle_cord_list=rectangle_cord_list,
                                                              all_vertices=all_vertices,
                                                              convex_cord=convex_cord,
                                                              concave_cord=concave_cord,
                                                              independent_chord=independent_chord,
                                                              nearest_chord=nearest_chord)

                    if not coordinate_statues:
                        rectangle_cord_list.pop()
                        # return False
                        continue
                    else:
                        return True

            # if it is upward direction, try whether the left bottom can meet the left upper coordinate
            else:

                # if ((candidate_coordinate in concave_cord) and (prev_cord in concave_cord)) or (
                #         [candidate_coordinate, prev_cord] in nearest_chord) or (
                #         [prev_cord, candidate_coordinate] in nearest_chord):
                #     if ([candidate_coordinate, prev_cord] in independent_chord) or (
                #             [prev_cord, candidate_coordinate] in independent_chord) or (
                #             [candidate_coordinate, prev_cord] in nearest_chord) or (
                #             [prev_cord, candidate_coordinate] in nearest_chord) or (
                #             [prev_cord, candidate_coordinate] in neighbour_list) or (
                #             [candidate_coordinate,prev_cord] in neighbour_list):
                if ((candidate_coordinate in concave_cord) and (last_cord in concave_cord)) or (
                        [candidate_coordinate, last_cord] in nearest_chord) or (
                        [last_cord, candidate_coordinate] in nearest_chord):
                    if ([candidate_coordinate, last_cord] in independent_chord) or (
                            [last_cord, candidate_coordinate] in independent_chord) or (
                            [candidate_coordinate, last_cord] in nearest_chord) or (
                            [last_cord, candidate_coordinate] in nearest_chord) or (
                            [last_cord, candidate_coordinate] in neighbour_list) or (
                            [candidate_coordinate, last_cord] in neighbour_list):
                        if candidate_coordinate == left_upper:
                            return True
                        else:
                            last_cord = candidate_coordinate
                            continue

                    else:
                        return False
                else:
                    if candidate_coordinate == left_upper:
                        return True
                    else:
                        last_cord = candidate_coordinate
                        continue

    return False


# if 2 convex coordinates are found check if there are any other colinear vertices between that prev coordinte
# and the candidate cordinate
def check_for_convex_coord_out_of_polygon(prev_cord, candidate_cord, candidate_list, all_vertices, independent_chord,
                                          nearest_chord):
    valid_cord = True
    for c_id, c in enumerate(candidate_list):

        if c == candidate_cord:
            continue
        else:
            # if c[0]==prev_cord[0] and c[0]==candidate_cord[0]:
            #     if (c[1] > prev_cord[1] and c[1] < candidate_cord[1]) :
            #         valid_cord = False
            #         break
            # elif c[1] == prev_cord[1] and c[1] == candidate_cord[1]:
            #     if c[1] > prev_cord[1] and c[1] < candidate_cord[1]:
            #         valid_cord = False
            #         break

            if c[0] == prev_cord[0] and c[0] == candidate_cord[0]:
                if (candidate_cord[1] > prev_cord[1] and candidate_cord[1] < c[1]) or (
                        candidate_cord[1] < prev_cord[1] and candidate_cord[1] > c[1]):
                    valid_cord = False
                    break
            elif c[1] == prev_cord[1] and c[1] == candidate_cord[1]:
                if (candidate_cord[0] > prev_cord[0] and candidate_cord[0] < c[0]) or (
                        candidate_cord[0] < prev_cord[0] and candidate_cord[0] > c[0]):
                    valid_cord = False
                    break
    # check if there is any valid chord between the candidate and the prev coord.
    # for

    # if valid_cord:


    return valid_cord


# based on the given direction an current coordination find the vertices which are inline with the current coord
# either vertically or horizontally.
def find_cordinates_based_on_direction(direction, current_cord, all_vertices):
    candidate_coord = []
    # finding coordinates in the rightward
    if direction == 1:  # left
        for vertices in all_vertices:
            if (vertices[0] == current_cord[0]) and (vertices[1] > current_cord[1]):
                candidate_coord.append(vertices)

        # sort according to increasing order towards horizontal direction
        candidate_coord.sort()

    # finding coordinates in the downward
    elif direction == 2:
        for vertices in all_vertices:
            if (vertices[1] == current_cord[1]) and (vertices[0] > current_cord[0]):
                candidate_coord.append(vertices)
        # sort according to increasing order towards vertical direction
        candidate_coord.sort()

    # finding coordinate in the leftward
    elif direction == 3:
        for vertices in all_vertices:
            if (vertices[0] == current_cord[0]) and (vertices[1] < current_cord[1]):
                candidate_coord.append(vertices)

        # sort according to the decreasing order of horizontal direction
        candidate_coord.sort(reverse=True)

    # finding coordinate in the upward
    else:
        for vertices in all_vertices:
            if (vertices[1] == current_cord[1]) and (vertices[0] < current_cord[0]):
                candidate_coord.append(vertices)

        # sort according to the decreasing oreder of vertical direction
        candidate_coord.sort(reverse=True)

    return candidate_coord


# given all the rectangles, there can be certain cases one reactangle is completely overlappin with a bigger rectangle
def remove_overlapping_rectangles(rectangles):
    ref_rect_ind = 0
    length_rectangle = len(rectangles) - 1
    # for ref_rect_ind in range(len(rectangles) - 1):
    while ref_rect_ind < length_rectangle - 1:
        ref_rect = rectangles[ref_rect_ind]
        validation_flags = []
        check_rect_ind = ref_rect_ind + 1
        while check_rect_ind <= length_rectangle:
            check_rect = rectangles[check_rect_ind]

            validation_flags = check_overlap(ref_rect, check_rect)

            # if checked rectangle is in the ref rectangle remove that rectangle from the list
            if validation_flags[0] == 1 and validation_flags[1] == -1:
                rectangles.pop(check_rect_ind)
                length_rectangle -= 1
                check_rect_ind -= 1

            check_rect_ind += 1

            if validation_flags[0] == -1:
                break

        if validation_flags[0] == -1:
            rectangles.pop(ref_rect_ind)
            length_rectangle -= 1
            ref_rect_ind -= 1
        ref_rect_ind += 1

    return rectangles


# check the overlapping of the rectangles.
# if the overlapping happens, state whether what is the biggest and smallest one.
#       1,-1 : overlapp happens. rect_2 is smaller
#       -1, 1 : overlap happens. rect_1 is smaller
# 0, 0 no overlap happens
def check_overlap(rect_1, rect_2):
    valid_rect_1 = 0
    valid_rect_2 = 0
    if (rect_1[0][0] <= rect_2[0][0]) and (rect_1[0][1] <= rect_2[0][1]) and (  # compare the left upper
            rect_1[1][0] <= rect_2[1][0]) and (rect_1[1][1] >= rect_2[1][1]) and (  # compare right upper
            rect_1[2][0] >= rect_2[2][0]) and (rect_1[2][1] >= rect_2[2][1]) and (  # comapre right lower
            rect_1[3][0] >= rect_2[3][0]) and (rect_1[3][1] <= rect_2[3][1]):  # compare the left lower

        # rect_1_size = (rect_1[1][0]-rect_1[0][0])*(rect_1[2][1]-rect_1[1][1])
        # rect_2_size = (rect_2[1][0] - rect_2[0][0]) * (rect_2[2][1] - rect_2[1][1])
        valid_rect_1 = 1
        valid_rect_2 = -1

    elif (rect_1[0][0] >= rect_2[0][0]) and (rect_1[0][1] >= rect_2[0][1]) and (  # compare the left upper
            rect_1[1][0] >= rect_2[1][0]) and (rect_1[1][1] <= rect_2[1][1]) and (  # compare right upper
            rect_1[2][0] <= rect_2[2][0]) and (rect_1[2][1] <= rect_2[2][1]) and (  # comapre right lower
            rect_1[3][0] <= rect_2[3][0]) and (rect_1[3][1] >= rect_2[3][1]):  # compare the left lower
        valid_rect_1 = -1
        valid_rect_2 = 1

    return valid_rect_1, valid_rect_2


# check if given coordinates are crossing a given chord derived
def cross_chords(prev_cord, candidate_coordinate, independent_chord, nearest_chord, neighbour_list):
    is_crossing_chord = False
    # check for independent chord
    all_chords = []
    for ind, chord in enumerate(independent_chord):
        all_chords.append(chord)
    for ind, chord in enumerate(nearest_chord):
        all_chords.append(chord)

    for i_chord_ind, i_chord in enumerate(all_chords):
        if prev_cord[0] == candidate_coordinate[0]:
            if i_chord[0][1] == i_chord[1][1]:
                # if ((prev_cord[0] < i_chord[0][0]) or (prev_cord[0] > i_chord[0][0])) and (
                #         (prev_cord[0] < i_chord[1][0]) or (prev_cord[0] > i_chord[1][0])) and (
                #         i_chord[0][1] > prev_cord[1] or i_chord[0][1] < prev_cord[1]) and (
                #         i_chord[0][1] > candidate_coordinate[1] or i_chord[0][1] < candidate_coordinate[1]):
                #     is_crossing_chord = True

                if (((prev_cord[0] < i_chord[0][0]) and (prev_cord[0] > i_chord[1][0])) or (
                        (prev_cord[0] > i_chord[0][0]) and (prev_cord[0] < i_chord[1][0]))) and (
                        ((i_chord[0][1] > prev_cord[1]) and (i_chord[1][1] < candidate_coordinate[1])) or (
                        (i_chord[0][1] < prev_cord[1]) and (i_chord[1][1] > candidate_coordinate[1]))):
                    is_crossing_chord = True

        else:
            if i_chord[0][0] == i_chord[1][0]:
                if (((prev_cord[0] < i_chord[0][0]) and (candidate_coordinate[0] > i_chord[0][0])) or (
                        (prev_cord[0] > i_chord[0][0]) and (candidate_coordinate[0] < i_chord[0][0]))) and (
                        ((i_chord[0][1] > prev_cord[1]) and (i_chord[1][1] < prev_cord[1])) or (
                        (i_chord[0][1] < prev_cord[1]) and (i_chord[1][1] > prev_cord[1]))):
                    is_crossing_chord = True

        # check if the given coordinate list is intersecting the boundaary line. Ideally all the boundary line should
        # represent by the neighbourhood list.
        # check whether if there are any attempt to cross the boundary line.
        for n_ind, n_pair in enumerate(neighbour_list):
            if (((prev_cord[0] < n_pair[0][0]) and (candidate_coordinate[0] > n_pair[0][0])) or (
                    (prev_cord[0] > n_pair[0][0]) and (candidate_coordinate[0] < n_pair[0][0]))) and (
                    ((n_pair[0][1] > prev_cord[1]) and (n_pair[1][1] < prev_cord[1])) or (
                    (n_pair[0][1] < prev_cord[1]) and (n_pair[1][1] > prev_cord[1]))):
                is_crossing_chord = True

    return is_crossing_chord

# def main():
#
#     return


# if __name__ == main():
#
#     main()
