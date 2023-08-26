import numpy as np
import stim
import networkx as nx
from itertools import combinations
from collections import deque


def bfs_shortest_paths(graph, start, end):
    """
    counts the number of shortest paths between a start node and an end node in an undirected graph that doesn't have weights
    """
    # Initialize a dictionary to keep track of visited nodes and their distances
    visited = {start: 0}
    # Initialize a dictionary to keep track of the number of shortest paths to each node
    shortest_paths = {start: 1}
    # Initialize a queue to store nodes to visit
    queue = deque([start])

    # Loop through the queue until it is empty or the end node is found
    while queue:
        # Get the next node to visit
        node = queue.popleft()
        # If the end node is found, return the number of shortest paths to it
        if node == end:
            return shortest_paths[node], visited[node]
        # Loop through the neighbors of the current node
        for neighbor in graph.neighbors(node):
            # If the neighbor has not been visited yet, add it to the visited dictionary and queue
            if neighbor not in visited:
                visited[neighbor] = visited[node] + 1
                shortest_paths[neighbor] = shortest_paths[node]
                queue.append(neighbor)
            # If the neighbor has already been visited and is at the same distance as the current node,
            # increment the number of shortest paths to that neighbor
            elif visited[neighbor] == visited[node] + 1:
                shortest_paths[neighbor] += shortest_paths[node]

    # If the end node was not found, return 0
    return 0, 0


class DataQubit:
    def __init__(self, name, coords) -> None:
        self.name = name
        self.coords = coords

    def __repr__(self) -> str:
        return f"{self.name}, Coords: {self.coords}"


class MeasureQubit:
    def __init__(self, name, coords, data_qubits, basis) -> None:
        self.name = name
        self.coords = coords
        self.data_qubits = data_qubits
        self.basis = basis

    def __repr__(self):
        return f"|{self.name}, Coords: {self.coords}, Basis: {self.basis}, Data Qubits: {self.data_qubits}|"


class Defect:
    # can be a cluster of defects
    def __init__(self, name, coords, x_gauges, z_gauges) -> None:
        self.name = name  # names of the disabled data qubits in the defect
        self.coords = coords  # coords of the disabled data qubits in the defect
        self.x_gauges = x_gauges  # list of MeasureQubit
        self.z_gauges = z_gauges  # list of MeasureQubit
        max_x = 0
        max_y = 0
        min_x = 9999
        min_y = 9999
        for x, y in coords:
            if x > max_x:
                max_x = x
            if x < min_x:
                min_x = x
            if y > max_y:
                max_y = y
            if y < min_y:
                min_y = y
        self.diameter = int(max(max_x - min_x, max_y - min_y) / 2 + 1)
        self.horizontal_len = int((max_x - min_x) / 2 + 1)
        self.vertical_len = int((max_y - min_y) / 2 + 1)

    def change_diameter(self, new_size):
        self.diameter = new_size

    def __repr__(self) -> str:
        return f"{self.name}, Coords: {self.coords}, Gauges: {self.gauges}"


class DisJointSets:
    def __init__(self, N):
        # Initially, all elements are single element subsets
        self._parents = [node for node in range(N)]
        self._ranks = [1 for _ in range(N)]

    def find(self, u):
        while u != self._parents[u]:
            # path compression technique
            self._parents[u] = self._parents[self._parents[u]]
            u = self._parents[u]
        return u

    def connected(self, u, v):
        return self.find(u) == self.find(v)

    def connected_components(self):
        components = [[0]]
        for i in range(1, len(self._parents)):
            new_component = True
            for component in components:
                if self.connected(component[0], i):
                    component.append(i)
                    new_component = False
                    break
            if new_component:
                components.append([i])
        return components

    def union(self, u, v):
        # Union by rank optimization
        root_u, root_v = self.find(u), self.find(v)
        if root_u == root_v:
            return True
        if self._ranks[root_u] > self._ranks[root_v]:
            self._parents[root_v] = root_u
        elif self._ranks[root_v] > self._ranks[root_u]:
            self._parents[root_u] = root_v
        else:
            self._parents[root_u] = root_v
            self._ranks[root_v] += 1
        return False


class LogicalQubit:
    def __init__(
        self,
        d: int,
        readout_err: float,
        gate1_err: float,
        gate2_err: float,
        missing_coords=[],
        data_loss_tolerance=1.0,
        verbose=False,
        get_metrics=False,
    ) -> None:
        self.d = d
        self.readout_err = readout_err
        self.gate1_err = gate1_err
        self.gate2_err = gate2_err
        self.percolated = False
        self.too_many_qubits_lost = False
        self.vertical_distance = 0
        self.horizontal_distance = 0
        self.num_inactive_data = 0
        self.num_inactive_syn = 0
        self.num_data_superstabilizer = 0
        # the data qubits on the corners and the weight-2 stabilizers that are attached to them
        # 4 corners [top left data, top left syn, top right data, top right syn,
        # bottom left data, bottom left syn, bottom right data, bottom right syn]
        corners = [
            0,
            d**2 + d // 2,
            d - 1,
            d**2 + d // 2 - 1,
            d**2 - d,
            2 * d**2 - 1 - d // 2,
            d**2 - 1,
            2 * d**2 - 2 - d // 2,
        ]
        # look up data qubit from coords
        data_matching = [[None for _ in range(2 * d)] for _ in range(2 * d)]
        # look up syndrome qubit name from coords
        syndrome_matching = {}
        # lookup table for qubit status, data qubits then syndromes
        is_disabled = np.zeros(
            2 * d**2 - 1
        )  # 0:normal, 1: disabled(requires superstabilizer), -1:deleted (due to defect on boundary)
        # all data qubits, including the missing ones. DataQubit objects
        data_list = [None for i in range(d**2)]
        # data qubits on the dynamic boundaries: red left, red right, blue top, blue bottom
        # note: they are either on the original boundary or next to a deleted qubit
        dynamic_boundaries = [[], [], [], []]
        # Defect(disabled_data_q.name if disabled_data_q else None, missing_coords, defect_gauges) #defines a cluster of defect
        self.defect = []  # an array of defect clusters

        # look up the data qubits connected to the syndrome qubits, including the disabled
        syn_info = []  # items are MeasureQubit
        # look up the syn qubits connected to the data qubits, including the disabled
        data_info = [[] for i in range(d**2)]  # items are lists of qubit names
        self.boundary_deformation = np.empty(
            4
        )  # total width of defects on each boundary (excluding ends), left:0,right:1,top:2,bottom:3

        for x in range(d):
            for y in range(d):
                name = d * x + y
                coords = (2 * x, 2 * y)
                # initialize the dynamic boundaries
                if x == 0:  # top edge
                    dynamic_boundaries[2].append(name)
                elif x == d - 1:  # bottom edge
                    dynamic_boundaries[3].append(name)
                if y == 0:  # left edge
                    dynamic_boundaries[0].append(name)
                elif y == d - 1:  # right edge
                    dynamic_boundaries[1].append(name)
                if coords in missing_coords:
                    is_disabled[name] = 1
                data_q = DataQubit(name, coords)
                data_matching[data_q.coords[0]][data_q.coords[1]] = data_q
                data_list[name] = data_q

        q = d * d
        for x in range(-1, d):
            for y in range(-1, d):
                if (x + y) % 2 == 1 and x != -1 and x != d - 1:  # is X syndrome
                    coords = (2 * x + 1, 2 * y + 1)
                    syndrome_matching[coords] = q
                    data_qubits = []
                    if y != d - 1:
                        data_qubits += [
                            data_matching[coords[0] + 1][coords[1] + 1],
                            data_matching[coords[0] - 1][coords[1] + 1],
                        ]
                    else:
                        data_qubits += [None, None]
                    if y != -1:
                        data_qubits += [
                            data_matching[coords[0] + 1][coords[1] - 1],
                            data_matching[coords[0] - 1][coords[1] - 1],
                        ]
                    else:
                        data_qubits += [None, None]
                    measure_q = MeasureQubit(q, coords, data_qubits, "X")
                    for data_q in data_qubits:
                        if data_q is not None:
                            data_info[data_q.name].append(q)
                    if coords in missing_coords:  # defective syndrome qubit
                        is_disabled[q] = 1
                    syn_info.append(measure_q)
                    q += 1
                elif (x + y) % 2 == 0 and y != -1 and y != d - 1:  # is Z syndrome
                    coords = (2 * x + 1, 2 * y + 1)
                    syndrome_matching[coords] = q
                    data_qubits = []
                    if x != d - 1:
                        data_qubits += [
                            data_matching[coords[0] + 1][coords[1] + 1],
                            data_matching[coords[0] + 1][coords[1] - 1],
                        ]
                    else:
                        data_qubits += [None, None]
                    if x != -1:
                        data_qubits += [
                            data_matching[coords[0] - 1][coords[1] + 1],
                            data_matching[coords[0] - 1][coords[1] - 1],
                        ]
                    else:
                        data_qubits += [None, None]
                    measure_q = MeasureQubit(q, coords, data_qubits, "Z")
                    for data_q in data_qubits:
                        if data_q is not None:
                            data_info[data_q.name].append(q)
                    if coords in missing_coords:  # defective syndrome qubit
                        is_disabled[q] = 1
                    syn_info.append(measure_q)
                    q += 1

        def check_remaining_qubits():  # returns True if too many qubits are lost
            self.num_inactive_data = np.count_nonzero(is_disabled[: d**2])
            self.num_inactive_syn = np.count_nonzero(is_disabled[d**2 :])
            self.num_data_superstabilizer = np.count_nonzero(is_disabled[: d**2] == 1)
            if self.num_inactive_data > d**2 * data_loss_tolerance:
                self.too_many_qubits_lost = True
                return True
            return False

        def is_boundary(qname):
            # return True if the qubit is on the (original) boundary
            if qname < d**2:  # data qubit
                x, y = data_list[qname].coords
                if x == 0 or x == 2 * (d - 1) or y == 0 or y == 2 * (d - 1):
                    return True
                else:
                    return False
            else:  # syndrome qubit
                x, y = syn_info[qname - d**2].coords
                if x == -1 or x == 2 * d - 1 or y == -1 or y == 2 * d - 1:
                    return True
                else:
                    return False

        def near_blue_boundary(qname):
            x, y = syn_info[qname - d**2].coords
            if x == 1 or x == 2 * d - 3:
                return True
            return False

        def near_red_boundary(qname):
            x, y = syn_info[qname - d**2].coords
            if y == 1 or y == 2 * d - 3:
                return True
            return False

        def near_boundary(qname):
            # syndrome qubit near the (original) boundary
            assert qname >= d**2
            x, y = syn_info[qname - d**2].coords
            if near_blue_boundary(qname) or near_red_boundary(qname):
                return True
            else:
                return False

        # handle the corners, if any of the 2 qubits at a corner is broken, delete that corner
        for i in range(4):
            if is_disabled[corners[2 * i]] or is_disabled[corners[2 * i + 1]]:
                is_disabled[corners[2 * i]] = -1
                is_disabled[corners[2 * i + 1]] = -1
                if verbose:
                    print("Delete corner:", corners[2 * i], corners[2 * i + 1])

        tentative_disable = []  # qubits disabled while handling interior defects
        # might need to relabel them as active if we deform the boundary again

        def handle_disconnected_syn(only_handle_boundary=False):
            # find syn connected to none or only 1 active data - disable the syn
            any_change = False
            for syn_q in syn_info:  # for all syndrome
                # if not already disabled
                if (not is_disabled[syn_q.name]) or (
                    only_handle_boundary and is_disabled[syn_q.name] == 1
                ):
                    active_data = []
                    near_deleted_qubit = False
                    for data_q in syn_q.data_qubits:
                        if data_q is not None:
                            if is_disabled[data_q.name] == 0 or (
                                only_handle_boundary and is_disabled[data_q.name] == 1
                            ):
                                active_data.append(data_q)
                            elif is_disabled[data_q.name] == -1:
                                near_deleted_qubit = True
                    if len(active_data) <= 1:
                        any_change = True
                        if only_handle_boundary:
                            if near_deleted_qubit:
                                is_disabled[
                                    syn_q.name
                                ] = -1  # delete the syndrome qubit
                                if verbose:
                                    print("Handle disconnected syndrome:")
                                    print("\t delete", syn_q.name)
                        else:
                            is_disabled[syn_q.name] = 1  # disable the syndrome qubit
                            tentative_disable.append(syn_q.name)
                            if verbose:
                                print("Handle disconnected syndrome:")
                                print("\t disable", syn_q.name)
                    elif (
                        len(active_data) == 2
                        and (not only_handle_boundary)
                        and is_disabled[syn_q.name] == 0
                    ):
                        # get rid of the thin weight-2 gauge
                        if (
                            active_data[0].coords[0] != active_data[1].coords[0]
                            and active_data[0].coords[1] != active_data[1].coords[1]
                        ):
                            any_change = True
                            is_disabled[syn_q.name] = 1
                            tentative_disable.append(syn_q.name)
                            if verbose:
                                print("Handle disconnected syndrome:")
                                print("\t disable", syn_q.name)
            return any_change  # if any new qubits are disabled

        def handle_disconnected_data(only_handle_boundary=False):
            # find data qubit measured by only X syn or only Z syn, or no syn - disable it
            any_change = False
            for i in range(d**2):
                if (not is_disabled[i]) or (
                    only_handle_boundary and is_disabled[i] == 1
                ):
                    active_xsyn = []
                    active_zsyn = []
                    near_deleted_qubit = False
                    for j in data_info[i]:  # get idx of a neighboring syndrome
                        # if the syn is active
                        if (not is_disabled[j]) or (
                            only_handle_boundary and is_disabled[j] == 1
                        ):
                            if syn_info[j - d**2].basis == "X":
                                active_xsyn.append(j)
                            else:
                                assert syn_info[j - d**2].basis == "Z"
                                active_zsyn.append(j)
                        elif is_disabled[j] == -1:  # if the syn is deleted
                            near_deleted_qubit = True
                    if len(active_xsyn) == 0 or len(active_zsyn) == 0:
                        if only_handle_boundary:
                            if near_deleted_qubit:
                                any_change = True
                                is_disabled[i] = -1  # delete the data qubit
                                if verbose:
                                    print("Handle disconnected data:")
                                    print("\t delete", i)
                        elif not only_handle_boundary:
                            any_change = True
                            is_disabled[i] = 1  # disable the data qubit
                            tentative_disable.append(i)
                            if verbose:
                                print("Handle disconnected data:")
                                print("\t disable", i)
            return any_change  # if any new qubits are disabled

        def remove_disconnected_component(only_handle_boundary=False):
            def not_missing(qname):
                if only_handle_boundary:
                    if is_disabled[qname] != -1:
                        return True
                elif is_disabled[qname] == 0:
                    return True
                return False

            active_data_q = [data_list[i] for i in range(d**2) if not_missing(i)]
            union_find = DisJointSets(len(active_data_q))
            for i in range(len(active_data_q)):
                for j in range(i + 1, len(active_data_q)):
                    syn_qubits_i = [
                        syn
                        for syn in data_info[active_data_q[i].name]
                        if not_missing(syn)
                    ]
                    syn_qubits_j = [
                        syn
                        for syn in data_info[active_data_q[j].name]
                        if not_missing(syn)
                    ]
                    if set(syn_qubits_i) & set(
                        syn_qubits_j
                    ):  # data qubits shared by a syndrome
                        union_find.union(i, j)
            # each item is an arr of indices in active_data_q
            connected_data = sorted(
                union_find.connected_components(), key=len, reverse=True
            )
            if len(connected_data) > 1:
                if verbose:
                    print("Multiple connected components, remove all but largest")
                for i in range(1, len(connected_data)):
                    for j in connected_data[i]:
                        if only_handle_boundary:
                            is_disabled[active_data_q[j].name] = -1
                            if verbose:
                                print("\t Remove", active_data_q[j].name)
                        else:
                            is_disabled[active_data_q[j].name] = 1
                            tentative_disable.append(active_data_q[j].name)
                            if verbose:
                                print("\t Disable", active_data_q[j].name)
                return True
            return False

        def handle_disconnected_qubits(only_handle_boundary=False):
            # combine handle_disconnected_syn() and handle_disconnected_data()
            any_change = False
            in_progress = True
            while in_progress:
                in_progress = False
                if handle_disconnected_syn(only_handle_boundary=only_handle_boundary):
                    in_progress = True
                    any_change = True
                if handle_disconnected_data(only_handle_boundary=only_handle_boundary):
                    in_progress = True
                    any_change = True
                if remove_disconnected_component(
                    only_handle_boundary=only_handle_boundary
                ):
                    in_progress = True
                    any_change = True
            return any_change

        def handle_boundary_data(
            qname,
        ):  # handle broken data on the original boundary by deleting 4 qubits
            weight4_syn = []
            weight2_syn = []
            for synq in data_info[qname]:  # syn connected to the data qubit
                if not is_boundary(synq):  # the syn is a weight-4 one
                    weight4_syn.append(synq)
                else:
                    weight2_syn.append(synq)
            assert len(weight4_syn) == 2
            assert len(weight2_syn) == 1
            is_disabled[weight2_syn[0]] = -1  # delete the weight-2 syn
            if verbose:
                print("Delete weight-2 syndrome qubit", weight2_syn[0])
            data_to_delete = []  # delete 2 data qubits
            for dataq in syn_info[weight2_syn[0] - d**2].data_qubits:
                if dataq is not None:
                    is_disabled[dataq.name] = -1
                    data_to_delete.append(dataq.name)
            if verbose:
                print("Delete data qubits", data_to_delete)
            neighbors_of_syn0 = syn_info[weight4_syn[0] - d**2].data_qubits
            assert None not in neighbors_of_syn0
            datanames_of_syn0 = [dataq.name for dataq in neighbors_of_syn0]
            if all(data_idx in datanames_of_syn0 for data_idx in data_to_delete):
                is_disabled[weight4_syn[0]] = -1
                if verbose:
                    print("Delete syndrome qubit", weight4_syn[0])
            else:
                is_disabled[weight4_syn[1]] = -1
                if verbose:
                    print("Delete syndrome qubit", weight4_syn[1])

        def handle_boundary_defects(qname):  # qname is the name of a broken qubit
            # if qname is on or near (the original) boundary, mark it as deleted
            # delete nearby qubits as needed
            # return if the qubit is handled - if it is not on the boundary it is not handled
            if qname < d**2:  # data qubit
                if is_boundary(qname):
                    handle_boundary_data(qname)
                    return True
                else:
                    return False
            else:  # syndrome qubit
                if is_boundary(qname):  # weight-2 syndrome
                    for data_idx in syn_info[qname - d**2].data_qubits:
                        if data_idx is not None:
                            handle_boundary_data(data_idx.name)  # delete 4 qubits
                            return True
                elif near_boundary(qname):  # weight-4 syn near boundary
                    # 2 cases: same color as boundary (hard)
                    # or different color than boundary (easy - delete 4 qubits)
                    # corner treated as different color from boundary
                    syn_basis = syn_info[qname - d**2].basis
                    x, y = syn_info[qname - d**2].coords
                    is_disabled[qname] = -1
                    if verbose:
                        print("Delete syndrome qubit", qname)
                    easy = False
                    if syn_basis == "X" and near_blue_boundary(qname):  # easy
                        if x == 1:
                            x1 = -1
                        else:
                            assert x == 2 * d - 3
                            x1 = 2 * d - 1
                        weight2_qname = syndrome_matching[
                            (x1, y)
                        ]  # name of the neighboring weight2 stabilizer
                        easy = True
                    elif syn_basis == "Z" and near_red_boundary(qname):  # easy
                        if y == 1:
                            y1 = -1
                        else:
                            assert y == 2 * d - 3
                            y1 = 2 * d - 1
                        weight2_qname = syndrome_matching[
                            (x, y1)
                        ]  # name of the neighboring weight2 stabilizer
                        easy = True
                    if easy:
                        is_disabled[weight2_qname] = -1
                        if verbose:
                            print("Delete weight-2 syndrome qubit", weight2_qname)
                        data_to_delete = []  # delete 2 data qubits
                        for dataq in syn_info[weight2_qname - d**2].data_qubits:
                            if dataq is not None:
                                is_disabled[dataq.name] = -1
                                data_to_delete.append(dataq.name)
                        if verbose:
                            print("Delete data qubits", data_to_delete)
                    else:  # hard
                        # nearby syndrome qubits to delete
                        if x == 1:  # upper edge
                            syn_to_delete = [
                                syndrome_matching[(x, y - 2)],
                                syndrome_matching[(x, y + 2)],
                                syndrome_matching[(x + 2, y)],
                                syndrome_matching[(x - 2, y - 2)],
                                syndrome_matching[(x - 2, y + 2)],
                            ]
                        elif x == 2 * d - 3:  # lower edge
                            syn_to_delete = [
                                syndrome_matching[(x, y - 2)],
                                syndrome_matching[(x, y + 2)],
                                syndrome_matching[(x - 2, y)],
                                syndrome_matching[(x + 2, y - 2)],
                                syndrome_matching[(x + 2, y + 2)],
                            ]
                        elif y == 1:  # left edge
                            syn_to_delete = [
                                syndrome_matching[(x - 2, y)],
                                syndrome_matching[(x + 2, y)],
                                syndrome_matching[(x, y + 2)],
                                syndrome_matching[(x - 2, y - 2)],
                                syndrome_matching[(x + 2, y - 2)],
                            ]
                        else:  # right edge
                            assert y == 2 * d - 3
                            syn_to_delete = [
                                syndrome_matching[(x - 2, y)],
                                syndrome_matching[(x + 2, y)],
                                syndrome_matching[(x, y - 2)],
                                syndrome_matching[(x - 2, y + 2)],
                                syndrome_matching[(x + 2, y + 2)],
                            ]
                        if verbose:
                            print("Handle syn near boundary (same color)")
                            print("\t delete syn qubits:", syn_to_delete)
                        for synq in syn_to_delete:
                            is_disabled[synq] = -1
                    return True
                else:  # not on the boundary so not handled
                    return False

        def count_remaining_dataq(qname):
            # return number of syndrome qubit qname's data qubits that have not been deleted
            assert qname >= d**2
            count = 0
            for dataq in syn_info[qname - d**2].data_qubits:
                if dataq is not None:
                    if is_disabled[dataq.name] != -1:
                        count += 1
            return count

        def handle_defect_on_new_boundary(qname):  # qname is name of a broken qubit
            # if qname is on or near (the dynamic) boundary, mark it as deleted
            # return if the qubit is handled - if it is not on or near the boundary it is not handled
            if qname < d**2:  # data qubit
                # data qubit on the boundary -> only next to 1 stabilizer whose color is different
                # from the boundary. delete that syndrome qubit.
                # first handle edge case - corner
                if (
                    qname == dynamic_boundaries[0][0]
                    or qname == dynamic_boundaries[0][-1]
                    or qname == dynamic_boundaries[1][0]
                    or qname == dynamic_boundaries[1][-1]
                ):
                    # special case: corner
                    # if qname is next to a weight-2 syndrome, delete that syndrome
                    # if it is not next to any weight-2 syndrome, delete the lower weight adjacent syndrome.
                    active_syn = []
                    for syn_name in data_info[
                        qname
                    ]:  # syndrome qubits connected to qname
                        if is_disabled[syn_name] != -1:
                            active_syn.append(syn_name)
                    assert len(active_syn) == 2
                    if (
                        is_disabled[active_syn[0]] == 1
                        or is_disabled[active_syn[1]] == 1
                    ):
                        if verbose:
                            print(
                                "Postpone handling corner defect",
                                qname,
                                "because it will be deleted when handling a neighbor syndrome defect",
                            )
                        return False
                    else:
                        if verbose:
                            print("Handle (new) corner defect", qname)
                        if count_remaining_dataq(active_syn[0]) < count_remaining_dataq(
                            active_syn[1]
                        ):
                            is_disabled[active_syn[0]] = -1
                            if verbose:
                                print("\t Also delete syndrome", active_syn[0])
                        else:
                            is_disabled[active_syn[1]] = -1
                            if verbose:
                                print("\t Also delete syndrome", active_syn[1])
                        is_disabled[qname] = -1
                        return True
                elif (
                    qname in dynamic_boundaries[0] or qname in dynamic_boundaries[1]
                ):  # red boundary
                    # delete the blue stabilizer that qname is adjacent to
                    is_disabled[qname] = -1
                    if verbose:
                        print("Delete defective data qubit on new red boundary:", qname)
                    for syn_name in data_info[
                        qname
                    ]:  # syndrome qubits connected to qname
                        if is_disabled[syn_name] != -1:  # exclude the deleted syndromes
                            if (
                                syn_info[syn_name - d**2].basis == "Z"
                            ):  # blue syndrome
                                is_disabled[syn_name] = -1
                                if verbose:
                                    print("\t also delete syndrome", syn_name)
                                return True
                elif (
                    qname in dynamic_boundaries[2] or qname in dynamic_boundaries[3]
                ):  # blue boundary
                    # delete the red stabilizer that qname is adjacent to
                    is_disabled[qname] = -1
                    if verbose:
                        print(
                            "Delete defective data qubit on new blue boundary:", qname
                        )
                    for syn_name in data_info[
                        qname
                    ]:  # syndrome qubits connected to qname
                        if is_disabled[syn_name] != -1:  # exclude the deleted syndromes
                            if syn_info[syn_name - d**2].basis == "X":  # red syndrome
                                is_disabled[syn_name] = -1
                                if verbose:
                                    print("\t also delete syndrome", syn_name)
                                return True
                else:
                    # check if the superstabilizer around the defect would require any deleted qubit
                    delete_basis = "N"  # None
                    for syn_name in data_info[qname]:
                        if count_remaining_dataq(syn_name) < 4:
                            if syn_info[syn_name - d**2].basis == "X":  # red boundary
                                # delete neighboring blue syndromes
                                delete_basis = "Z"
                            else:  # blue boundary
                                # delete neighboring red syndromes
                                delete_basis = "X"
                            break
                    if delete_basis == "N":
                        return False
                    is_disabled[qname] = -1
                    if verbose:
                        print(
                            "Delete defective data qubit",
                            qname,
                            "(superstablizer requires deleted qubits)",
                        )
                    for syn_name in data_info[qname]:
                        if syn_info[syn_name - d**2].basis == delete_basis:
                            is_disabled[syn_name] = -1
                            if verbose:
                                print("\t Delete syndrome qubit", syn_name)
                    return True
            else:  # syndrome qubit
                # syndrome qubit is on the boundary if one of its data qubits is on the boundary
                boundary_color = "N"  # None
                for dataq in syn_info[qname - d**2].data_qubits:
                    if dataq is not None:
                        if (
                            dataq.name == dynamic_boundaries[0][0]
                            or dataq.name == dynamic_boundaries[0][-1]
                            or dataq.name == dynamic_boundaries[1][0]
                            or dataq.name == dynamic_boundaries[1][-1]
                        ):
                            # data qubit at a corner
                            is_disabled[qname] = -1
                            is_disabled[dataq.name] = -1
                            if verbose:
                                print(
                                    "Delete (new) corner syndrome",
                                    qname,
                                    "and data qubit",
                                    dataq.name,
                                )
                            return True
                        elif (
                            dataq.name in dynamic_boundaries[0]
                            or dataq.name in dynamic_boundaries[1]
                        ):
                            boundary_color = "X"
                            break
                        elif (
                            dataq.name in dynamic_boundaries[2]
                            or dataq.name in dynamic_boundaries[3]
                        ):
                            boundary_color = "Z"
                            break
                if boundary_color == "N":  # not on the boundary
                    return False
                is_disabled[qname] = -1
                # if it is on a boundary with a different color - easy case, just delete it
                if syn_info[qname - d**2].basis != boundary_color:
                    if verbose:
                        print("Delete (new) boundary syndrome", qname)
                else:  # if it is only on a boundary with the same color - hard case,
                    #  also need to delete adjacent syndromes of different color
                    x, y = syn_info[qname - d**2].coords
                    # coords of neighboring syndromes with different colors
                    possible_neighbor_coords = [
                        (x, y + 2),
                        (x, y - 2),
                        (x + 2, y),
                        (x - 2, y),
                    ]
                    to_delete = []
                    for coords in possible_neighbor_coords:
                        if coords in syndrome_matching:
                            syn_name = syndrome_matching[coords]
                            if is_disabled[syn_name] != -1:
                                to_delete.append(syn_name)
                                is_disabled[syn_name] = -1
                    if verbose:
                        print("Handle defective (new) boundary syndrome", qname)
                        print("\t also delete", to_delete)
                return True

        def neighbors_on_boundary(qname, boundary_type):
            # return a list data qubit names, which are input data qubit's neighbors on a boundary
            # colors = "red" or "blue"
            x0, y0 = data_list[qname].coords
            neighbors = []
            # syndromes of the specified color that the given qubit is adjacent to
            active_syn_of_color = []
            for syn_name in data_info[qname]:  # get idx of a neighboring syndrome
                if is_disabled[syn_name] != -1:  # syndrome is not deleted
                    if syn_info[syn_name - d**2].basis == boundary_type:
                        active_syn_of_color.append(syn_name)
            for syn_name in active_syn_of_color:
                remaining_dataq = []
                deleted_dataq = []
                for dataq in syn_info[syn_name - d**2].data_qubits:
                    if dataq is not None:
                        if is_disabled[dataq.name] == -1:
                            deleted_dataq.append(dataq)
                        else:
                            remaining_dataq.append(dataq)
                if len(remaining_dataq) == 2:
                    if remaining_dataq[0].name == qname:
                        neighbors.append(remaining_dataq[1].name)
                    else:
                        neighbors.append(remaining_dataq[0].name)
                elif len(remaining_dataq) == 3:
                    assert len(deleted_dataq) == 1
                    for dataq in remaining_dataq:
                        x1, y1 = dataq.coords
                        if abs(x1 - x0) == 2 and abs(y1 - y0) == 2:
                            neighbors.append(dataq.name)
                            break
                else:
                    assert len(remaining_dataq) == 4
                    syn_x, syn_y = syn_info[syn_name - d**2].coords
                    neighbor_syn0 = (syn_x, syn_y + 2 * (y0 - syn_y))
                    neighbor_data0 = (x0 + 2 * (syn_x - x0), y0)
                    neighbor_syn1 = (syn_x + 2 * (x0 - syn_x), syn_y)
                    neighbor_data1 = (x0, y0 + 2 * (syn_y - y0))
                    if (
                        neighbor_syn0 in syndrome_matching
                        and is_disabled[syndrome_matching[neighbor_syn0]] == -1
                    ) or neighbor_syn0 not in syndrome_matching:
                        neighbors.append(
                            data_matching[neighbor_data0[0]][neighbor_data0[1]].name
                        )
                    if (
                        neighbor_syn1 in syndrome_matching
                        and is_disabled[syndrome_matching[neighbor_syn1]] == -1
                    ) or neighbor_syn1 not in syndrome_matching:
                        neighbors.append(
                            data_matching[neighbor_data1[0]][neighbor_data1[1]].name
                        )
            return neighbors

        def extend_boundary_list(boundary_list, boundary_type):
            # print('extend boundary list, start with', boundary_list)
            # recursive function that extends the list of boundary data qubits
            if (
                boundary_list[0] == -1 and boundary_list[-1] == -1
            ):  # both ends have terminated
                extended_boundary = boundary_list[
                    1:-1
                ]  # get rid of the -1 at both ends
                return extended_boundary
            else:
                extended_boundary = boundary_list[:]  # copy the list
                if boundary_list[0] != -1:  # extend at beginning
                    neighbors = neighbors_on_boundary(boundary_list[0], boundary_type)
                    if len(neighbors) == 1:  # only 1 neighbor - hits an end
                        if len(boundary_list) == 1:
                            extended_boundary.insert(
                                0, -1
                            )  # mark the front as complete
                            extended_boundary.append(neighbors[0])
                        else:  # otherwise that neighbor is already in boundary_list
                            extended_boundary.insert(
                                0, -1
                            )  # mark the front as complete
                    else:
                        assert len(neighbors) == 2 and neighbors[0] != neighbors[1]
                        if len(boundary_list) == 1:
                            extended_boundary.insert(0, neighbors[0])
                            extended_boundary.append(neighbors[1])
                        else:
                            if neighbors[0] == boundary_list[1]:
                                extended_boundary.insert(0, neighbors[1])
                            else:
                                assert neighbors[1] == boundary_list[1]
                                extended_boundary.insert(0, neighbors[0])
                if len(boundary_list) > 1 and boundary_list[-1] != -1:  # extend at end
                    neighbors = neighbors_on_boundary(boundary_list[-1], boundary_type)
                    if len(neighbors) == 1:  # only 1 neighbor - hits an end
                        assert neighbors[0] == boundary_list[-2]
                        extended_boundary.append(-1)  # mark the end as complete
                    else:
                        assert len(neighbors) == 2
                        if neighbors[0] == boundary_list[-2]:
                            extended_boundary.append(neighbors[1])
                        else:
                            assert neighbors[1] == boundary_list[-2]
                            extended_boundary.append(neighbors[0])
                return extend_boundary_list(extended_boundary, boundary_type)

        def update_dynamic_boundary():  # keep track of the boundaries
            # return whether percolation has happened
            for i in range(4):
                # when there's a change in the boundary, find the new boundary from a data qubit in
                # the old boundary that is still present
                unchanged_nodes = []  # indices of unchanged nodes in the list
                for j in range(len(dynamic_boundaries[i])):
                    if is_disabled[dynamic_boundaries[i][j]] != -1:
                        unchanged_nodes.append(j)
                if len(unchanged_nodes) == len(dynamic_boundaries[i]):
                    continue  # no change in this boundary
                elif len(unchanged_nodes) == 0:
                    err_messaege = "An entire boundary has shifted:" + str(i)
                    raise RuntimeError(err_messaege)

                new_boundary = []
                unchanged_node = dynamic_boundaries[i][
                    unchanged_nodes[0]
                ]  # the first unchanged node
                if i <= 1:
                    boundary_type = "X"
                else:
                    boundary_type = "Z"
                new_boundary = extend_boundary_list([unchanged_node], boundary_type)
                # decide whether to reverse the boundary qubit list
                x_start, y_start = data_list[new_boundary[0]].coords
                x_end, y_end = data_list[new_boundary[-1]].coords
                if (i <= 1 and x_start > x_end) or (i > 1 and y_start > y_end):
                    new_boundary.reverse()
                dynamic_boundaries[i] = new_boundary
            if verbose:
                print("New boundary:", dynamic_boundaries)
            # check if the new boundaries are valid
            # 2 red boundaries or 2 blue boundaries touch each other -> invalid
            if (
                len(set(dynamic_boundaries[0]).intersection(set(dynamic_boundaries[1])))
                > 0
            ):
                if verbose:
                    print(
                        "The red boundaries collapse at",
                        set(dynamic_boundaries[0]).intersection(
                            set(dynamic_boundaries[1])
                        ),
                    )
                return True
            if (
                len(set(dynamic_boundaries[2]).intersection(set(dynamic_boundaries[3])))
                > 0
            ):
                if verbose:
                    print(
                        "The blue boundaries collapse at",
                        set(dynamic_boundaries[2]).intersection(
                            set(dynamic_boundaries[3])
                        ),
                    )
                return True
            # red and blue boundaries don't connect -> invalid
            # top left, bottom left, top right, bottom right
            if (
                dynamic_boundaries[0][0] != dynamic_boundaries[2][0]
                or dynamic_boundaries[0][-1] != dynamic_boundaries[3][0]
                or dynamic_boundaries[2][-1] != dynamic_boundaries[1][0]
                or dynamic_boundaries[1][-1] != dynamic_boundaries[3][-1]
            ):
                if verbose:
                    print("The boundaries do not connect")
                return True
            return False

        def handle_broken_syn():
            # disable data qubits connected to broken syndrome qubits
            any_change = False
            for i in range(len(syn_info)):
                if is_disabled[i + d**2] == 1:  # if the syndrome qubit is disabled
                    for data_q in syn_info[i].data_qubits:
                        if data_q is not None:
                            if is_disabled[data_q.name] == 0:
                                is_disabled[data_q.name] = 1
                                tentative_disable.append(data_q.name)
                                any_change = True
                                if verbose:
                                    print(
                                        "Handle broken syndrome",
                                        i + d**2,
                                        ": disable",
                                        data_q.name,
                                    )
            return any_change  # if any new qubits are disabled

        # continue to deform the boundary as needed, mark qubits as deleted.

        # first handle the special case: syndromes near boundary with same color than the boundary
        # and is close enough to a corner
        # lower left,lower right,upper left,upper right
        special_cases = [
            syndrome_matching[(2 * d - 3, 3)],
            syndrome_matching[(2 * d - 5, 2 * d - 3)],
            syndrome_matching[(3, 1)],
            syndrome_matching[(1, 2 * d - 5)],
        ]
        if is_disabled[special_cases[0]] == 1:  # lower left
            if (
                is_disabled[syndrome_matching[(2 * d - 3, 1)]] == 0
                and is_disabled[syndrome_matching[(2 * d - 3, 5)]] == 0
                and is_disabled[syndrome_matching[(2 * d - 5, 3)]] == 0
            ):
                is_disabled[special_cases[0]] = -1
                is_disabled[data_matching[2 * d - 2][0].name] = -1
                is_disabled[data_matching[2 * d - 2][2].name] = -1
                is_disabled[syndrome_matching[(2 * d - 1, 1)]] = -1
                if verbose:
                    print(
                        "special case, delete",
                        special_cases[0],
                        data_matching[2 * d - 2][0].name,
                        data_matching[2 * d - 2][2].name,
                        syndrome_matching[(2 * d - 1, 1)],
                    )
        if is_disabled[special_cases[1]] == 1:  # lower right
            if (
                is_disabled[syndrome_matching[(2 * d - 3, 2 * d - 3)]] == 0
                and is_disabled[syndrome_matching[(2 * d - 5, 2 * d - 5)]] == 0
                and is_disabled[syndrome_matching[(2 * d - 7, 2 * d - 3)]] == 0
            ):
                is_disabled[special_cases[1]] = -1
                is_disabled[data_matching[2 * d - 4][2 * d - 2].name] = -1
                is_disabled[data_matching[2 * d - 2][2 * d - 2].name] = -1
                is_disabled[syndrome_matching[(2 * d - 3, 2 * d - 1)]] = -1
                if verbose:
                    print(
                        "special case, delete",
                        special_cases[1],
                        data_matching[2 * d - 4][2 * d - 2].name,
                        data_matching[2 * d - 2][2 * d - 2].name,
                        syndrome_matching[(2 * d - 3, 2 * d - 1)],
                    )
        if is_disabled[special_cases[2]] == 1:  # upper left
            if (
                is_disabled[syndrome_matching[(1, 1)]] == 0
                and is_disabled[syndrome_matching[(5, 1)]] == 0
                and is_disabled[syndrome_matching[(3, 3)]] == 0
            ):
                is_disabled[special_cases[2]] = -1
                is_disabled[data_matching[0][0].name] = -1
                is_disabled[data_matching[2][0].name] = -1
                is_disabled[syndrome_matching[(1, -1)]] = -1
                if verbose:
                    print(
                        "special case, delete",
                        special_cases[2],
                        data_matching[0][0].name,
                        data_matching[2][0].name,
                        syndrome_matching[(1, -1)],
                    )
        if is_disabled[special_cases[3]] == 1:  # upper right
            if (
                is_disabled[syndrome_matching[(1, 2 * d - 3)]] == 0
                and is_disabled[syndrome_matching[(1, 2 * d - 7)]] == 0
                and is_disabled[syndrome_matching[(3, 2 * d - 5)]] == 0
            ):
                is_disabled[special_cases[3]] = -1
                is_disabled[data_matching[0][2 * d - 4].name] = -1
                is_disabled[data_matching[0][2 * d - 2].name] = -1
                is_disabled[syndrome_matching[(-1, 2 * d - 3)]] = -1
                if verbose:
                    print(
                        "special case, delete",
                        special_cases[3],
                        data_matching[0][2 * d - 4].name,
                        data_matching[0][2 * d - 2].name,
                        syndrome_matching[(-1, 2 * d - 3)],
                    )

        boundary_invalid = update_dynamic_boundary()
        if boundary_invalid:
            self.percolated = True
            return
        # handle the defects on or near the original boundaries (corners already handled), including
        # data qubits on the edges, weight-2 syn and outer circle of weight-4 syn
        for i in range(len(is_disabled)):
            # this ensures the qubit is not at a corner and not already deleted
            if is_disabled[i] == 1:
                if verbose:
                    print("looking at defect", i)
                assert i not in corners
                if handle_boundary_defects(i):
                    handle_disconnected_qubits(
                        only_handle_boundary=True
                    )  # only handle boundary
                    if check_remaining_qubits():
                        return
                    boundary_invalid = update_dynamic_boundary()
                    if boundary_invalid:
                        self.percolated = True
                        return

        # handle defects on the new boundary until the boundaries are stable
        change = True
        while change:
            change = False
            for i in range(len(is_disabled)):
                if is_disabled[i] == 1:  # defective qubit that has not been deleted
                    if verbose:
                        print("looking at defect", i)
                    if handle_defect_on_new_boundary(i):
                        change = True
                        handle_disconnected_qubits(
                            only_handle_boundary=True
                        )  # only handle boundary
                        if check_remaining_qubits():
                            return
                        boundary_invalid = update_dynamic_boundary()
                        if boundary_invalid:
                            self.percolated = True
                            return

        # handle defects in the interior
        if verbose:
            print("Move on to the interior defects")
        # the faulty qubits currently in the interior. other qubits with disabled[i] = 1 are only tentatively disabled
        interior_defects = [i for i in range(len(is_disabled)) if is_disabled[i] == 1]

        def reset_tentative_disable():  # reset the list of tentatively disabled qubits and the interior defects
            new_interior_defects = [
                qname for qname in interior_defects if is_disabled[qname] != -1
            ]
            interior_defects[:] = new_interior_defects
            for qname in tentative_disable:
                if is_disabled[qname] != -1:
                    is_disabled[qname] = 0
            tentative_disable.clear()
            in_progress = True
            while in_progress:  # fill out the tentative_disable list
                in_progress = False
                if handle_broken_syn():  # only called here
                    in_progress = True
                if (
                    handle_disconnected_qubits()
                ):  # only called here with only_handle_boundary=False
                    in_progress = True

        reset_tentative_disable()

        change = True
        while change:  # further deform the boundary as needed
            change = False
            for i in interior_defects + tentative_disable:
                if is_disabled[i] == 1:
                    if handle_defect_on_new_boundary(i):
                        change = True
                        handle_disconnected_qubits(
                            only_handle_boundary=True
                        )  # only handle boundary
                        if check_remaining_qubits():
                            return
                        boundary_invalid = update_dynamic_boundary()
                        if boundary_invalid:
                            self.percolated = True
                            return
                        break
            if change:
                reset_tentative_disable()

        if check_remaining_qubits():
            return

        # check edge deformation
        temp = -2  # record position of last regular boundary qubit
        acc = 0
        across_defect = False
        for qname in dynamic_boundaries[0]:  # left edge
            x, y = data_list[qname].coords
            if y == 0:  # this is a regular boundary qubit
                if across_defect:
                    assert x - temp > 0
                    acc += (x - temp - 2) / 2
                    across_defect = False
                    if verbose:
                        print("Left edge, deformed between", temp, x)
                temp = x
            else:  # not a regular boundary qubit
                if temp != -2:
                    across_defect = True
        self.boundary_deformation[0] = acc
        acc = 0
        temp = -2
        across_defect = False
        for qname in dynamic_boundaries[1]:  # right edge
            x, y = data_list[qname].coords
            if y == 2 * (d - 1):  # this is a regular boundary qubit
                if across_defect:
                    assert x - temp > 0
                    acc += (x - temp - 2) / 2
                    across_defect = False
                    if verbose:
                        print("Right edge, deformed between", temp, x)
                temp = x
            else:  # not a regular boundary qubit
                if temp != -2:
                    across_defect = True
        self.boundary_deformation[1] = acc
        acc = 0
        temp = -2
        across_defect = False
        for qname in dynamic_boundaries[2]:  # top edge
            x, y = data_list[qname].coords
            if x == 0:  # this is a regular boundary qubit
                if across_defect:
                    assert y - temp > 0
                    acc += (y - temp - 2) / 2
                    across_defect = False
                    if verbose:
                        print("Top edge, deformed between", temp, y)
                temp = y
            else:  # not a regular boundary qubit
                if temp != -2:
                    across_defect = True
        self.boundary_deformation[2] = acc
        acc = 0
        temp = -2
        across_defect = False
        for qname in dynamic_boundaries[3]:  # bottom edge
            x, y = data_list[qname].coords
            if x == 2 * (d - 1):  # this is a regular boundary qubit
                if across_defect:
                    assert y - temp > 0
                    acc += (y - temp - 2) / 2
                    across_defect = False
                    if verbose:
                        print("Bottom edge, deformed between", temp, y)
                temp = y
            else:  # not a regular boundary qubit
                if temp != -2:
                    across_defect = True
        self.boundary_deformation[3] = acc

        if check_remaining_qubits():
            return

        # fill out the fields
        self.data = []
        for i in range(len(data_list)):
            if not is_disabled[i]:
                self.data.append(data_list[i])

        # partition the defects into clusters
        disabled_data_q = [data_list[i] for i in range(d**2) if is_disabled[i] == 1]
        if len(disabled_data_q) > 0:
            union_find = DisJointSets(len(disabled_data_q))
            for i in range(len(disabled_data_q)):
                for j in range(i + 1, len(disabled_data_q)):
                    ci = disabled_data_q[i].coords
                    cj = disabled_data_q[j].coords
                    # adjacent data qubits (including diagonal)
                    if abs(ci[0] - cj[0]) <= 2 and abs(ci[1] - cj[1]) <= 2:
                        union_find.union(i, j)
            # each item is an arr of indices in disabled_data_q
            connected_defects = union_find.connected_components()
            # the x and z gauges, partitioned according to connected_defects
            cluster_xgauge = [[] for cluster in connected_defects]
            cluster_zgauge = [[] for cluster in connected_defects]
        else:
            connected_defects = []

        # the stabilizers
        self.x_ancilla = []
        self.z_ancilla = []
        # the broken stabilizers (gauges)
        self.x_gauges = []
        self.z_gauges = []

        for syn_q in syn_info:
            if not is_disabled[syn_q.name]:
                new_data_qubits = []
                is_gauge = False
                disabled_neighbor_data = -1  # record the name of 1 broken data qubit
                for data_q in syn_q.data_qubits:  # data qubits connected to the syn
                    if data_q is None or is_disabled[data_q.name] == -1:
                        new_data_qubits.append(None)
                    else:
                        if is_disabled[data_q.name] == 1:  # if the data q is disabled
                            is_gauge = True
                            disabled_neighbor_data = data_q.name
                            new_data_qubits.append(None)
                        else:  # if the data q still exists
                            new_data_qubits.append(data_q)
                measure_q = MeasureQubit(
                    syn_q.name, syn_q.coords, new_data_qubits, syn_q.basis
                )
                if is_gauge:  # gauge operator
                    # identify which cluster it belongs to
                    for cluster_i in range(len(connected_defects)):
                        # disabled data qubits in the cluster
                        data_names = [
                            disabled_data_q[i].name
                            for i in connected_defects[cluster_i]
                        ]
                        assert disabled_neighbor_data != -1
                        if disabled_neighbor_data in data_names:
                            # then this gauge is in the cluster
                            if syn_q.basis == "X":
                                self.x_gauges.append(measure_q)
                                cluster_xgauge[cluster_i].append(measure_q)
                            else:
                                assert syn_q.basis == "Z"
                                self.z_gauges.append(measure_q)
                                cluster_zgauge[cluster_i].append(measure_q)
                else:  # stabilizer
                    if syn_q.basis == "X":
                        self.x_ancilla.append(measure_q)
                    else:
                        assert syn_q.basis == "Z"
                        self.z_ancilla.append(measure_q)

        for cluster_i in range(len(connected_defects)):
            data_names = [disabled_data_q[i].name for i in connected_defects[cluster_i]]
            data_coords = [
                disabled_data_q[i].coords for i in connected_defects[cluster_i]
            ]
            new_cluster = Defect(
                data_names,
                data_coords,
                cluster_xgauge[cluster_i],
                cluster_zgauge[cluster_i],
            )
            self.defect.append(new_cluster)
        # indices of all active qubits
        self.all_qubits = [i for i in range(len(is_disabled)) if not is_disabled[i]]

        if verbose:  # print out deleted qubits due to boundary defects
            deleted_qubits = [
                i for i in range(len(is_disabled)) if is_disabled[i] == -1
            ]
            print("List of deleted qubits due to boundary defects:")
            print(deleted_qubits)

        if get_metrics:
            # find shortest path from top boundary to bottom boundary
            Gx = nx.Graph()
            Gx.add_nodes_from([dataq.name for dataq in self.data])
            Gx.add_nodes_from([-1, -2])  # -1: source (top), -2: target (bottom)
            # connect 2 data qubits if they are in the same regular X stabilizer
            for syn_q in self.x_ancilla:
                active_data = [
                    dataq.name for dataq in syn_q.data_qubits if dataq is not None
                ]
                data_pairs = list(combinations(active_data, 2))
                Gx.add_edges_from(data_pairs)
            # connect 2 data qubits if they are in the same X super-stabilizer
            for defect in self.defect:
                active_data = set()
                for syn_q in defect.x_gauges:
                    for dataq in syn_q.data_qubits:
                        if dataq is not None:
                            active_data.add(dataq.name)
                data_pairs = list(combinations(active_data, 2))
                Gx.add_edges_from(data_pairs)
            # connect -1 to data qubits in the top boundary and -2 to data qubits in the bottom boundary
            for q in dynamic_boundaries[2]:
                Gx.add_edge(-1, q)
            for q in dynamic_boundaries[3]:
                Gx.add_edge(-2, q)
            try:
                # pathx = nx.shortest_path(Gx, source=-1, target=-2)
                num_shortest_paths, len_shortest_path = bfs_shortest_paths(Gx, -1, -2)
                self.all_shortest_paths_v = num_shortest_paths
                self.vertical_distance = len_shortest_path - 1
            except:
                raise RuntimeError("Failed to find vertical path")
            # if verbose:
            #    print("Shortest vertical logical operator",pathx[1:-1])
            # self.vertical_distance = len(pathx) - 2

            # find shortest path from left boundary to right boundary
            Gz = nx.Graph()
            Gz.add_nodes_from([dataq.name for dataq in self.data])
            Gz.add_nodes_from([-1, -2])  # -1: source (left), -2: target (right)
            # connect 2 data qubits if they are in the same regular Z stabilizer
            for syn_q in self.z_ancilla:
                active_data = [
                    dataq.name for dataq in syn_q.data_qubits if dataq is not None
                ]
                data_pairs = list(combinations(active_data, 2))
                Gz.add_edges_from(data_pairs)
            # connect 2 data qubits if they are in the same Z super-stabilizer
            for defect in self.defect:
                active_data = set()
                for syn_q in defect.z_gauges:
                    for dataq in syn_q.data_qubits:
                        if dataq is not None:
                            active_data.add(dataq.name)
                data_pairs = list(combinations(active_data, 2))
                Gz.add_edges_from(data_pairs)
            # connect -1 to data qubits in the left boundary and -2 to data qubits in the right boundary
            for q in dynamic_boundaries[0]:
                Gz.add_edge(-1, q)
            for q in dynamic_boundaries[1]:
                Gz.add_edge(-2, q)
            try:
                # pathz = nx.shortest_path(Gz, source=-1, target=-2)
                """#shortest paths by nx
                all_shortest_paths = nx.all_shortest_paths(Gz,-1,-2)
                count_shortest_paths = 0
                for _ in all_shortest_paths:
                    count_shortest_paths += 1
                self.all_shortest_paths = count_shortest_paths
                print("nx")
                """
                # """ #num of shortest path by BFS
                num_shortest_paths, len_shortest_path = bfs_shortest_paths(Gz, -1, -2)
                self.all_shortest_paths_h = num_shortest_paths
                self.horizontal_distance = len_shortest_path - 1
                # """
            except:
                raise RuntimeError("Failed to find horizontal path")
            # self.horizontal_distance = len(pathz) - 2
            # if verbose:
            #    print("Shortest horizontal logical operator",pathz[1:-1])

        # find the observable. Shortest path from top to bottom that doesn't any red stabilizer
        # It cannot cross any red gauge!
        G_obs = nx.Graph()
        G_obs.add_nodes_from([dataq.name for dataq in self.data])
        G_obs.add_nodes_from([-1, -2])  # -1: source (top), -2: target (bottom)
        # connect 2 data qubits if they are in the same regular X stabilizer
        for syn_q in self.x_ancilla:
            active_data = [
                dataq.name for dataq in syn_q.data_qubits if dataq is not None
            ]
            data_pairs = list(combinations(active_data, 2))
            G_obs.add_edges_from(data_pairs)
        # connect 2 data qubits if they are in the same X gauge
        for defect in self.defect:
            for syn_q in defect.x_gauges:
                active_data = []
                for dataq in syn_q.data_qubits:
                    if dataq is not None:
                        active_data.append(dataq.name)
                data_pairs = list(combinations(active_data, 2))
                G_obs.add_edges_from(data_pairs)
        # connect -1 to data qubits in the top boundary and -2 to data qubits in the bottom boundary
        for q in dynamic_boundaries[2]:
            G_obs.add_edge(-1, q)
        for q in dynamic_boundaries[3]:
            G_obs.add_edge(-2, q)
        try:
            path_obs = nx.shortest_path(G_obs, source=-1, target=-2)
        except:
            raise RuntimeError("Failed to find observable")
        self.observable = path_obs[1:-1]
        if verbose:
            print("Observable:", self.observable)

        self.meas_record = []

    def num_disabled_qubits(self):
        return self.num_inactive_data + self.num_inactive_syn

    def num_disabled_data(self):
        return self.num_inactive_data

    def num_data_in_superstabilizer(self):
        return self.num_data_superstabilizer

    def num_disabled_syndromes(self):
        return self.num_inactive_syn

    def actual_distance_vertical(self):
        return self.vertical_distance

    def actual_distance_horizontal(self):
        return self.horizontal_distance

    def edge_deformation(self):
        # if len(self.deformed_boundaries) == 0:
        #    return 0
        # if len(self.deformed_boundaries) == 1:
        #    return 1
        # if len(self.deformed_boundaries) == 2:
        #    if (self.deformed_boundaries[0] + self.deformed_boundaries[1]) % 2 == 1:
        #        return 3
        #    else:
        #        return 2
        # if len(self.deformed_boundaries) == 3:
        #    return 4
        # return 5
        return self.boundary_deformation

    def print_gauges(self):
        # for debugging: print out the gauges
        i = 0
        if len(self.defect) == 0:
            print("No gauges")
        for cluster in self.defect:
            assert len(cluster.x_gauges) > 0 and len(cluster.z_gauges) > 0
            print("cluster:", i, " diameter=", cluster.diameter)
            print("Data qubits in the defect cluster:", cluster.name)
            defect_x_gauges = [xsyn.name for xsyn in cluster.x_gauges]
            defect_z_gauges = [zsyn.name for zsyn in cluster.z_gauges]
            print("X gauges:", defect_x_gauges)
            print("Z gauges:", defect_z_gauges)
            i += 1
        return

    def need_to_rerun(self):
        # contains the type of superstabilizer that should have been disallowed
        for cluster in self.defect:
            for syn in cluster.x_gauges + cluster.z_gauges:
                active_coords = []
                for data in syn.data_qubits:
                    if data is not None:
                        active_coords.append(data.coords)
                if len(active_coords) == 2:
                    if (
                        active_coords[0][0] != active_coords[1][0]
                        and active_coords[0][1] != active_coords[1][1]
                    ):
                        # diagonal
                        return True
        return False

    def diameter_biggest_cluster(self):
        max_diameter = 0
        for cluster in self.defect:
            if cluster.diameter > max_diameter:
                max_diameter = cluster.diameter
        return max_diameter

    def num_clusters(self):
        return len(self.defect)

    def terminated_due_to_qubit_loss(self):
        return self.too_many_qubits_lost

    def is_percolated(self):
        return self.percolated

    def change_shell_diameter(self, new_size):
        # change the cluster "diameter" value to adjust size of shells
        for cluster in self.defect:
            cluster.change_diameter(new_size)

    def reset_err_rates(self, readout_err, gate1_err, gate2_err):
        self.readout_err = readout_err
        self.gate1_err = gate1_err
        self.gate2_err = gate2_err

    def apply_1gate(self, circ, gate, qubits):
        circ.append(gate, qubits)
        circ.append("DEPOLARIZE1", self.all_qubits, self.gate1_err)
        circ.append("TICK")

    def apply_2gate(self, circ, gate, qubits):
        circ.append(gate, qubits)
        circ.append("DEPOLARIZE2", qubits, self.gate2_err)
        # apply 1Q depolarizing errors on idle qubits
        if len(qubits) < len(self.all_qubits):
            idle_qubits = list(set(self.all_qubits) - set(qubits))
            circ.append("DEPOLARIZE1", idle_qubits, self.gate1_err)
        circ.append("TICK")

    def reset_meas_qubits(self, circ, op, qubits, last=False):
        if op == "R":
            circ.append(op, qubits)
        circ.append("X_ERROR", qubits, self.readout_err)
        if op == "M" or op == "MR":  # MR is not used in the current version
            circ.append(op, qubits)
            # Update measurement record indices
            meas_round = {}
            for i in range(len(qubits)):
                q = qubits[-1 - i]
                meas_round[q] = -1 - i
            for round in self.meas_record:
                for q, idx in round.items():
                    round[q] = idx - len(qubits)
            self.meas_record.append(meas_round)
        if not last and len(qubits) < len(self.all_qubits):
            idle_qubits = list(set(self.all_qubits) - set(qubits))
            circ.append("DEPOLARIZE1", idle_qubits, self.gate1_err)

    def get_meas_rec(self, round_idx, qubit_name):
        return stim.target_rec(self.meas_record[round_idx][qubit_name])

    def syndrome_round(self, circ: stim.Circuit, first=False, double=False) -> None:
        # measure X gauges first, Z gauges next
        syn_except_xgauge = (
            [gauge.name for gauge in self.z_gauges]
            + [measure.name for measure in self.x_ancilla]
            + [measure.name for measure in self.z_ancilla]
        )
        syn_except_zgauge = (
            [gauge.name for gauge in self.x_gauges]
            + [measure.name for measure in self.x_ancilla]
            + [measure.name for measure in self.z_ancilla]
        )
        if first:
            self.reset_meas_qubits(circ, "R", self.all_qubits)
        else:
            self.reset_meas_qubits(circ, "R", syn_except_xgauge)
        circ.append("TICK")
        self.apply_1gate(
            circ,
            "H",
            [measure.name for measure in self.x_ancilla]
            + [x_gauge.name for x_gauge in self.x_gauges],
        )

        for i in range(4):
            err_qubits = []
            for measure_x in self.x_ancilla + self.x_gauges:
                if measure_x.data_qubits[i] != None:
                    err_qubits += [measure_x.name, measure_x.data_qubits[i].name]
            for measure_z in self.z_ancilla:
                if measure_z.data_qubits[i] != None:
                    err_qubits += [measure_z.data_qubits[i].name, measure_z.name]
            self.apply_2gate(circ, "CX", err_qubits)

        self.apply_1gate(
            circ,
            "H",
            [measure.name for measure in self.x_ancilla]
            + [x_gauge.name for x_gauge in self.x_gauges],
        )

        self.reset_meas_qubits(circ, "M", syn_except_zgauge)

        if not first:
            circ.append("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
            for ancilla in self.x_ancilla + self.z_ancilla:
                circ.append(
                    "DETECTOR",
                    [
                        self.get_meas_rec(-1, ancilla.name),
                        self.get_meas_rec(-2, ancilla.name),
                    ],
                    ancilla.coords + (0,),
                )
            for cluster in self.defect:
                if len(cluster.x_gauges) > 0:
                    defect_x_gauges = [
                        self.get_meas_rec(-1, xsyn.name) for xsyn in cluster.x_gauges
                    ] + [self.get_meas_rec(-3, xsyn.name) for xsyn in cluster.x_gauges]
                    circ.append("DETECTOR", defect_x_gauges, cluster.coords[0] + (0,))
        else:  # first round
            for ancilla in self.z_ancilla:
                circ.append(
                    "DETECTOR",
                    self.get_meas_rec(-1, ancilla.name),
                    ancilla.coords + (0,),
                )
        circ.append("TICK")
        if not double:
            return circ  # skip 2nd half
        self.reset_meas_qubits(circ, "R", syn_except_zgauge)
        circ.append("TICK")
        self.apply_1gate(circ, "H", [measure.name for measure in self.x_ancilla])

        for i in range(4):
            err_qubits = []
            for measure_x in self.x_ancilla:
                if measure_x.data_qubits[i] != None:
                    err_qubits += [measure_x.name, measure_x.data_qubits[i].name]
            for measure_z in self.z_ancilla + self.z_gauges:
                if measure_z.data_qubits[i] != None:
                    err_qubits += [measure_z.data_qubits[i].name, measure_z.name]
            self.apply_2gate(circ, "CX", err_qubits)

        self.apply_1gate(circ, "H", [measure.name for measure in self.x_ancilla])
        self.reset_meas_qubits(circ, "M", syn_except_xgauge)
        circ.append("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
        for ancilla in self.x_ancilla + self.z_ancilla:
            circ.append(
                "DETECTOR",
                [
                    self.get_meas_rec(-1, ancilla.name),
                    self.get_meas_rec(-2, ancilla.name),
                ],
                ancilla.coords + (0,),
            )

        if not first:  # not the first round of Z gauge measurements
            for cluster in self.defect:
                if len(cluster.z_gauges) > 0:
                    defect_z_gauges = [
                        self.get_meas_rec(-1, zsyn.name) for zsyn in cluster.z_gauges
                    ] + [self.get_meas_rec(-3, zsyn.name) for zsyn in cluster.z_gauges]
                    circ.append("DETECTOR", defect_z_gauges, cluster.coords[0] + (0,))
        else:  # the first round of Z gauge measurements
            for cluster in self.defect:
                if len(cluster.z_gauges) > 0:
                    defect_z_gauges = [
                        self.get_meas_rec(-1, zsyn.name) for zsyn in cluster.z_gauges
                    ]
                    circ.append("DETECTOR", defect_z_gauges, cluster.coords[0] + (0,))
        circ.append("TICK")
        return circ

    def generate_stim(self, rounds) -> stim.Circuit:
        all_data = [data.name for data in self.data]
        circ = stim.Circuit()

        # Coords
        for data in self.data:
            circ.append("QUBIT_COORDS", data.name, data.coords)
        for x_ancilla in self.x_ancilla + self.x_gauges:
            circ.append("QUBIT_COORDS", x_ancilla.name, x_ancilla.coords)
        for z_ancilla in self.z_ancilla + self.z_gauges:
            circ.append("QUBIT_COORDS", z_ancilla.name, z_ancilla.coords)

        if len(self.x_gauges) > 0:  # if there are superstabilizers
            self.syndrome_round(circ, first=True, double=True)
            circ.append(
                stim.CircuitRepeatBlock(
                    rounds - 1, self.syndrome_round(stim.Circuit(), double=True)
                )
            )
        else:  # if there aren't superstabilizers
            self.syndrome_round(circ, first=True)
            circ.append(
                stim.CircuitRepeatBlock(rounds - 1, self.syndrome_round(stim.Circuit()))
            )

        self.reset_meas_qubits(circ, "M", all_data, last=True)
        circ.append("SHIFT_COORDS", [], [0.0, 0.0, 1.0])

        for ancilla in self.z_ancilla:
            circ.append(
                "DETECTOR",
                [
                    self.get_meas_rec(-1, data.name)
                    for data in ancilla.data_qubits
                    if data is not None
                ]
                + [self.get_meas_rec(-2, ancilla.name)],
                ancilla.coords + (0,),
            )

        for cluster in self.defect:
            if len(cluster.z_gauges) > 0:
                z_gauge_data_names = []
                for z_gauge in cluster.z_gauges:
                    for data in z_gauge.data_qubits:
                        if data is not None:
                            z_gauge_data_names.append(data.name)
                z_data = [self.get_meas_rec(-1, name) for name in z_gauge_data_names]
                defect_z_gauges = [
                    self.get_meas_rec(-2, zsyn.name) for zsyn in cluster.z_gauges
                ]
                circ.append(
                    "DETECTOR", z_data + defect_z_gauges, cluster.coords[0] + (0,)
                )

        circ.append(
            "OBSERVABLE_INCLUDE",
            [self.get_meas_rec(-1, data_name) for data_name in self.observable],
            0,
        )

        return circ

    def qubits_to_measure(self, round: int, only_x=False, only_z=False):
        # for the shell option
        # return an array of the qubits to measured in the specified round
        # only_x: only return the X stabilizers and gauges to be measured in the round
        # only_z: only return the Z stabilizers and gauges to be measured in the round
        to_measure = []
        if not only_z:
            to_measure += self.x_ancilla
        if not only_x:
            to_measure += self.z_ancilla
        for cluster in self.defect:
            if (round // cluster.diameter) % 2 == 0:
                # measure the X gauges in this cluster
                if not only_z:
                    to_measure += cluster.x_gauges
            else:  # measure the Z gauges in this cluster
                if not only_x:
                    to_measure += cluster.z_gauges
        return to_measure

    def syndrome_round_shell(self, circ: stim.Circuit, round: int) -> None:
        if round == 0:
            self.reset_meas_qubits(circ, "R", self.all_qubits)
        else:
            # rest the qubits measured last round
            self.reset_meas_qubits(
                circ, "R", [q.name for q in self.qubits_to_measure(round - 1)]
            )
        circ.append("TICK")
        self.apply_1gate(
            circ, "H", [q.name for q in self.qubits_to_measure(round, only_x=True)]
        )

        for i in range(4):
            err_qubits = []
            for measure_x in self.qubits_to_measure(round, only_x=True):
                if measure_x.data_qubits[i] != None:
                    err_qubits += [measure_x.name, measure_x.data_qubits[i].name]
            for measure_z in self.qubits_to_measure(round, only_z=True):
                if measure_z.data_qubits[i] != None:
                    err_qubits += [measure_z.data_qubits[i].name, measure_z.name]
            self.apply_2gate(circ, "CX", err_qubits)

        self.apply_1gate(
            circ, "H", [q.name for q in self.qubits_to_measure(round, only_x=True)]
        )

        self.reset_meas_qubits(
            circ, "M", [q.name for q in self.qubits_to_measure(round)]
        )

        if round == 0:  # first round
            for ancilla in self.z_ancilla:
                circ.append(
                    "DETECTOR",
                    self.get_meas_rec(-1, ancilla.name),
                    ancilla.coords + (0,),
                )
            # for cluster in self.defect:
            #    for x_gauge in cluster.x_gauges:
            #        circ.append("DETECTOR", self.get_meas_rec(-1, x_gauge.name), x_gauge.coords + (0,))
        else:
            circ.append("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
            for ancilla in self.x_ancilla + self.z_ancilla:
                circ.append(
                    "DETECTOR",
                    [
                        self.get_meas_rec(-1, ancilla.name),
                        self.get_meas_rec(-2, ancilla.name),
                    ],
                    ancilla.coords + (0,),
                )
            for cluster in self.defect:
                if (round // cluster.diameter) % 2 == 0:  # measure x gauges
                    # define the individual X gauge operators
                    if round % cluster.diameter > 0:  # the gauges are not random
                        for x_gauge in cluster.x_gauges:
                            circ.append(
                                "DETECTOR",
                                [
                                    self.get_meas_rec(-1, x_gauge.name),
                                    self.get_meas_rec(-2, x_gauge.name),
                                ],
                                x_gauge.coords + (0,),
                            )
                    # X superstabilizer is defined in the 1st of X measurement, every 2*diameter rounds
                    elif round % cluster.diameter == 0:
                        defect_x_gauges = [
                            self.get_meas_rec(-1, xsyn.name)
                            for xsyn in cluster.x_gauges
                        ] + [
                            self.get_meas_rec(-1 * (cluster.diameter + 2), xsyn.name)
                            for xsyn in cluster.x_gauges
                        ]
                        circ.append(
                            "DETECTOR", defect_x_gauges, cluster.coords[0] + (0,)
                        )
                else:  # measure z gauges
                    if round % cluster.diameter > 0:  # the gauges are not random
                        for z_gauge in cluster.z_gauges:
                            # define the individual Z gauge operators
                            circ.append(
                                "DETECTOR",
                                [
                                    self.get_meas_rec(-1, z_gauge.name),
                                    self.get_meas_rec(-2, z_gauge.name),
                                ],
                                z_gauge.coords + (0,),
                            )
                    # Z superstabilizer is defined in the 1st round of Z measurement, every 2*diameter rounds
                    elif round % cluster.diameter == 0:
                        if (
                            round == cluster.diameter
                        ):  # first def of the Z superstabilizer
                            defect_z_gauges = [
                                self.get_meas_rec(-1, zsyn.name)
                                for zsyn in cluster.z_gauges
                            ]
                        else:
                            defect_z_gauges = [
                                self.get_meas_rec(-1, zsyn.name)
                                for zsyn in cluster.z_gauges
                            ] + [
                                self.get_meas_rec(
                                    -1 * (cluster.diameter + 2), zsyn.name
                                )
                                for zsyn in cluster.z_gauges
                            ]
                        circ.append(
                            "DETECTOR", defect_z_gauges, cluster.coords[0] + (0,)
                        )
        circ.append("TICK")
        return circ

    def last_z_round(self, total_rounds, cluster_diameter, superstabilizer=False):
        # return the idx of the last round where the Z gauges in the cluster are measured
        # the last of the (total_rounds)th round is returned as -2
        offset = total_rounds % (2 * cluster_diameter)
        if offset > cluster_diameter:  # the last, incomplete round ends with a Z round
            return -2  # point to the last round
        else:  # no incomplete round, or the incomplete round does not include any Z round
            return -2 - offset  # point to end of the previous round

    def generate_stim_shell(self, rounds, overide_max_diameter=0) -> stim.Circuit:
        # Note: overide_max_diameter must be >= the actual max_diameter
        all_data = [data.name for data in self.data]
        circ = stim.Circuit()

        # Coords
        for data in self.data:
            circ.append("QUBIT_COORDS", data.name, data.coords)
        for x_ancilla in self.x_ancilla + self.x_gauges:
            circ.append("QUBIT_COORDS", x_ancilla.name, x_ancilla.coords)
        for z_ancilla in self.z_ancilla + self.z_gauges:
            circ.append("QUBIT_COORDS", z_ancilla.name, z_ancilla.coords)
        # diameter of the largest cluster
        cluster_diameters = [cluster.diameter for cluster in self.defect]
        if overide_max_diameter == 0:
            max_diameter = max(cluster_diameters)
        else:
            max_diameter = overide_max_diameter
        # if all cluster diameters divide the largest one, use Repeat block
        if all(max_diameter % diameter == 0 for diameter in cluster_diameters):
            for i in range(max_diameter * 2):
                self.syndrome_round_shell(circ, i)
            assert rounds > 2
            circ_repeat = stim.Circuit()
            for i in range(max_diameter * 2, max_diameter * 4):
                self.syndrome_round_shell(circ_repeat, i)
            circ.append(stim.CircuitRepeatBlock(rounds - 1, circ_repeat))
        else:
            for i in range(max_diameter * 2 * rounds):
                self.syndrome_round_shell(circ, i)

        self.reset_meas_qubits(circ, "M", all_data, last=True)
        circ.append("SHIFT_COORDS", [], [0.0, 0.0, 1.0])

        for ancilla in self.z_ancilla:  # Z stabilizers
            circ.append(
                "DETECTOR",
                [
                    self.get_meas_rec(-1, data.name)
                    for data in ancilla.data_qubits
                    if data is not None
                ]
                + [self.get_meas_rec(-2, ancilla.name)],
                ancilla.coords + (0,),
            )

        for cluster in self.defect:  # Z gauges
            # if the cluster ends in a Z round, define detectors for individual gauges
            # if the cluster ends in an X round, define Z superstabilizer and connect to the last Z round
            assert len(cluster.z_gauges) > 0
            z_cluster_data_names = []
            z_gauge_round = self.last_z_round(
                2 * max_diameter * rounds, cluster.diameter
            )

            for z_gauge in cluster.z_gauges:
                z_gauge_data_names = []
                for data in z_gauge.data_qubits:
                    if data is not None:
                        z_gauge_data_names.append(data.name)
                z_cluster_data_names += z_gauge_data_names
                if z_gauge_round == -2:  # if the last round is a Z round
                    z_data_gauge = [
                        self.get_meas_rec(-1, name) for name in z_gauge_data_names
                    ]
                    circ.append(
                        "DETECTOR",
                        z_data_gauge + [self.get_meas_rec(z_gauge_round, z_gauge.name)],
                        z_gauge.coords + (0,),
                    )
            if z_gauge_round != -2:  # if the last round is an X round
                # z superstabilizer from the last z round
                defect_z_gauges = [
                    self.get_meas_rec(z_gauge_round, zsyn.name)
                    for zsyn in cluster.z_gauges
                ]
                # z superstabilizer calculated from the data qubit readouts at the end
                z_data_cluster = [
                    self.get_meas_rec(-1, name) for name in z_cluster_data_names
                ]
                circ.append(
                    "DETECTOR",
                    z_data_cluster + defect_z_gauges,
                    cluster.coords[0] + (0,),
                )

        circ.append(
            "OBSERVABLE_INCLUDE",
            [self.get_meas_rec(-1, data_name) for data_name in self.observable],
            0,
        )

        return circ
