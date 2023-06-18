"""
Created by Malena Díaz Río. 
"""
#get the xmin, xmax, ymin, ymac of each annotated cycle 
def get_cycle_annotations(json):
    cicles = json['cardiac_cycles']
    limits_x = []
    limits_y = []
    for cycle in cicles:
        cycle_points_x = []
        cycle_points_y = []
        for point in cycle["control_points"]:
            cycle_points_x.append(point["x"])
            cycle_points_y.append(point["y"])
        limits_x.append([min(cycle_points_x), max(cycle_points_x)])
        limits_y.append([min(cycle_points_y), max(cycle_points_y)])
    return limits_x, limits_y, json['label']