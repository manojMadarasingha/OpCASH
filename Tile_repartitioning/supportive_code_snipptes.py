# this class contains the supportive functions to develop the algorithms

# this functions retursn the chords which are connected
# hole -- boundary line
# hole --
def find_chords_combined_with_holes(holes, concave_inds, horizontal_chords, vertical_chords, neighbour_lists):
    # holes.append([[6, 8], [6, 9], [7, 9], [7, 8]])

    for hole in holes:
        for hole_v_ind, hole_v in enumerate(hole):

            matching_coord_hori = []
            matching_coord_verti = []

            # ====== left upper coordinte of the hole starts========#
            if hole_v_ind == 0:

                max_col_ind = -1
                max_row_ind = -1

                # check for any matching horizontal coordinates
                # check for boundary coordinates
                for concave_ind in concave_inds:
                    if concave_ind[0] == hole_v[0]:
                        if concave_ind[1] < hole_v[1]:
                            if concave_ind[1] > max_col_ind:
                                max_col_ind = concave_ind[1]
                                matching_coord_hori = concave_ind
                # check for any hole coordinate

                for other_holes in holes:
                    if other_holes == hole:
                        continue
                    else:
                        # check if the other hole is at the left of the reffering hole
                        if other_holes[1][1] < hole_v[1]:
                            # are there any holes at the same level of the considered holes. This can be either upper ow lower
                            # edge of that particular hole
                            if other_holes[1][0] == hole_v[0]:
                                # choosing the closest other hole point near to the extracted hole
                                if other_holes[1][1] > max_col_ind:
                                    max_col_ind = other_holes[1][1]
                                    matching_coord_hori = other_holes[1]

                            elif other_holes[2][0] == hole_v[0]:
                                # choosing the closest other hole point near to the extracted hole
                                if other_holes[2][1] > max_col_ind:
                                    max_col_ind = other_holes[2][1]
                                    matching_coord_hori = other_holes[2]
                            ####### for the moment uncomment this part as such occurence would be rarely observable
                            # # check for any perpendicular boundary line which crosses the chords between the 2 holes
                            #     for neighbour_pairs in neighbour_lists:
                            #         if other_holes[1][0] == hole_v[0]:
                            #             check_if_chord_intersects([other_holes[1], hole_v], neighbour_pairs)
                            #         else:
                            #             check_if_chord_intersects([other_holes[2], hole_v], neighbour_pairs
                # check for any matching vertical coordinates on the boundary
                for concave_ind in concave_inds:
                    if concave_ind[1] == hole_v[1]:
                        if concave_ind[0] < hole_v[0]:
                            if concave_ind[0] > max_row_ind:
                                max_row_ind = concave_ind[0]
                                matching_coord_verti = concave_ind

                # check for any matching vertical coordianate with another hole above the given hole.

                for other_holes in holes:
                    if other_holes == hole:
                        continue
                    else:
                        # check if the other hole is at the upper of the reffering hole
                        if other_holes[3][0] < hole_v[0]:
                            # are there any holes at the same level of the considered holes. This can be either upper ow lower
                            # edge of that particular hole
                            if other_holes[3][1] == hole_v[1]:
                                # choosing the closest other hole point near to the extracted hole
                                if other_holes[3][0] > max_row_ind:
                                    max_row_ind = other_holes[3][0]
                                    matching_coord_verti = other_holes[3]

                            elif other_holes[2][1] == hole_v[1]:
                                # choosing the closest other hole point near to the extracted hole
                                if other_holes[2][0] > max_row_ind:
                                    max_row_ind = other_holes[2][0]
                                    matching_coord_verti = other_holes[2]
            # ====== left upper coordinte of the hole ends========#

            # ====== right upper coordinte of the hole starts=====#
            if hole_v_ind == 1:  # left upper coordinte of the hole

                min_col_ind = 100
                max_row_ind = -1

                # check for any matching horizontal coordinates
                # check for boundary coordinates right of the hole
                for concave_ind in concave_inds:
                    if concave_ind[0] == hole_v[0]:
                        if concave_ind[1] > hole_v[1]:
                            if concave_ind[1] < min_col_ind:
                                min_col_ind = concave_ind[1]
                                matching_coord_hori = concave_ind
                # check for any hole coordinate

                for other_holes in holes:
                    if other_holes == hole:
                        continue
                    else:
                        # check if the other hole is at the right of the reffering hole
                        if other_holes[0][1] > hole_v[1]:
                            # are there any holes at the same level of the considered holes. This can be either upper ow lower
                            # edge of that particular hole
                            if other_holes[0][0] == hole_v[0]:
                                if other_holes[0][0] < min_col_ind:
                                    min_col_ind = other_holes[0][0]
                                    matching_coord_hori = other_holes[0]

                            elif other_holes[3][0] == hole_v[0]:
                                if other_holes[3][0] < min_col_ind:
                                    min_col_ind = other_holes[3][0]
                                    matching_coord_hori = other_holes[3]

                # check for any matching vertical coordinates on the boundary
                for concave_ind in concave_inds:
                    if concave_ind[1] == hole_v[1]:
                        if concave_ind[0] < hole_v[0]:
                            if concave_ind[0] > max_row_ind:
                                max_row_ind = concave_ind[0]
                                matching_coord_verti = concave_ind

                # check for any matching vertical coordianate with another hole above the given hole.
                for other_holes in holes:
                    if other_holes == hole:
                        continue
                    else:
                        if other_holes[3][0] < hole_v[0]:
                            # are there any holes at the same level of the considered holes. This can be either upper ow lower
                            # edge of that particular hole
                            if other_holes[3][1] == hole_v[1]:
                                # choosing the closest other hole point near to the extracted hole
                                if other_holes[3][0] > max_row_ind:
                                    max_row_ind = other_holes[3][0]
                                    matching_coord_verti = other_holes[3]

                            elif other_holes[2][1] == hole_v[1]:
                                # choosing the closest other hole point near to the extracted hole
                                if other_holes[2][0] > max_row_ind:
                                    max_row_ind = other_holes[2][0]
                                    matching_coord_verti = other_holes[2]
            # ====== right upper coordinte of the hole ends========#

            # ====== right lower coordinte of the hole starts======#
            if hole_v_ind == 2:  # left upper coordinte of the hole

                min_col_ind = 100
                min_row_ind = 100

                # check for any matching horizontal coordinates
                # check for boundary coordinates right of the hole
                for concave_ind in concave_inds:
                    if concave_ind[0] == hole_v[0]:
                        if concave_ind[1] > hole_v[1]:
                            if concave_ind[1] < min_col_ind:
                                min_col_ind = concave_ind[1]
                                matching_coord_hori = concave_ind

                # check for any hole coordinate
                for other_holes in holes:
                    if other_holes == hole:
                        continue
                    else:
                        # check if the other hole is at the right of the reffering hole
                        if other_holes[0][1] > hole_v[1]:
                            # are there any holes at the same level of the considered holes. This can be either upper ow lower
                            # edge of that particular hole
                            if other_holes[0][0] == hole_v[0]:
                                if other_holes[0][0] < min_col_ind:
                                    min_col_ind = other_holes[0][0]
                                    matching_coord_hori = other_holes[0]

                            elif other_holes[3][0] == hole_v[0]:
                                if other_holes[3][0] < min_col_ind:
                                    min_col_ind = other_holes[3][0]
                                    matching_coord_hori = other_holes[3]

                # check for any matching vertical coordinates on the boundary
                for concave_ind in concave_inds:
                    if concave_ind[1] == hole_v[1]:
                        if concave_ind[0] > hole_v[0]:
                            if concave_ind[0] < min_row_ind:
                                min_row_ind = concave_ind[0]
                                matching_coord_verti = concave_ind

                # check for any matching vertical coordianate with another hole above the given hole.
                for other_holes in holes:
                    if other_holes == hole:
                        continue
                    else:
                        if other_holes[1][0] > hole_v[0]:
                            # are there any holes at the same level of the considered holes. This can be either upper ow lower
                            # edge of that particular hole
                            if other_holes[0][1] == hole_v[1]:
                                # choosing the closest other hole point near to the extracted hole
                                if other_holes[0][0] < min_row_ind:
                                    min_row_ind = other_holes[0][0]
                                    matching_coord_verti = other_holes[0]

                            elif other_holes[1][1] == hole_v[1]:
                                # choosing the closest other hole point near to the extracted hole
                                if other_holes[1][0] < min_row_ind:
                                    min_row_ind = other_holes[1][0]
                                    matching_coord_verti = other_holes[1]
            # ====== right upper coordinte of the hole ends========#

            # ====== left lower coordinte of the hole starts========#
            if hole_v_ind == 3:  # left upper coordinte of the hole

                max_col_ind = -1
                min_row_ind = 100

                # check for any matching horizontal coordinates
                # check for boundary coordinates
                for concave_ind in concave_inds:
                    if concave_ind[0] == hole_v[0]:
                        if concave_ind[1] < hole_v[1]:
                            if concave_ind[1] > max_col_ind:
                                max_col_ind = concave_ind[1]
                                matching_coord_hori = concave_ind
                # check for any hole coordinate

                for other_holes in holes:
                    if other_holes == hole:
                        continue
                    else:
                        # check if the other hole is at the left of the reffering hole
                        if other_holes[1][1] < hole_v[1]:
                            # are there any holes at the same level of the considered holes. This can be either upper ow lower
                            # edge of that particular hole
                            if other_holes[1][0] == hole_v[0]:
                                # choosing the closest other hole point near to the extracted hole
                                if other_holes[1][1] > max_col_ind:
                                    max_col_ind = other_holes[1][1]
                                    matching_coord_hori = other_holes[1]

                            elif other_holes[2][0] == hole_v[0]:
                                # choosing the closest other hole point near to the extracted hole
                                if other_holes[2][1] > max_col_ind:
                                    max_col_ind = other_holes[2][1]
                                    matching_coord_hori = other_holes[2]

                # check for any matching vertical coordinates on the boundary
                for concave_ind in concave_inds:
                    if concave_ind[1] == hole_v[1]:
                        if concave_ind[0] > hole_v[0]:
                            if concave_ind[0] < min_row_ind:
                                min_row_ind = concave_ind[0]
                                matching_coord_verti = concave_ind

                # check for any matching vertical coordianate with another hole above the given hole.
                for other_holes in holes:
                    if other_holes == hole:
                        continue
                    else:
                        if other_holes[0][0] > hole_v[0]:
                            # are there any holes at the same level of the considered holes. This can be either upper ow lower
                            # edge of that particular hole
                            if other_holes[0][1] == hole_v[1]:
                                # choosing the closest other hole point near to the extracted hole
                                if other_holes[0][0] < min_row_ind:
                                    min_row_ind = other_holes[0][0]
                                    matching_coord_verti = other_holes[0]

                            elif other_holes[1][1] == hole_v[1]:
                                # choosing the closest other hole point near to the extracted hole
                                if other_holes[1][0] < min_row_ind:
                                    min_row_ind = other_holes[1][0]
                                    matching_coord_verti = other_holes[1]

            # ====== left lower coordinte of the hole starts========#
            if len(matching_coord_verti) > 0:
                if not (([hole_v, list(matching_coord_verti)] in vertical_chords) or (
                        [list(matching_coord_verti), hole_v] in vertical_chords)):
                    vertical_chords.append([hole_v, list(matching_coord_verti)])
            if len(matching_coord_hori) > 0:
                if not (([hole_v, list(matching_coord_hori)] in horizontal_chords) or (
                        [list(matching_coord_hori), hole_v] in horizontal_chords)):
                    horizontal_chords.append([hole_v, list(matching_coord_hori)])

    return horizontal_chords, vertical_chords

