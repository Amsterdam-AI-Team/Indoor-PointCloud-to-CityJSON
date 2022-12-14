"""
CityJSON tools for polygon meshes.

The documentation of CityJSON can be found here:
https://www.cityjson.org/
"""
import os
import json
import logging
import numpy as np


logger = logging.getLogger(__name__)

def add_point_set():
    # # room coordinates to vertices
    # coords = [point for room in rooms for primitive in room for point in primitive]

    # # create vertices
    # vertices, inv_vertices = np.unique(coords, axis=0, return_inverse=True)
    # cityjson['vertices'] = vertices.tolist()

    # # convert to wavefront type format
    # v_index = inv_vertices.tolist()
    # for i in range(len(rooms)):
    #     outer_shell = [[[v_index.pop(0) for x in primitive]] for primitive in rooms[i]]
    #     building_part = {
    #         "type": "BuildingPart",
    #         "parents": [parent_id],
    #         "geometry": [{
    #             "type": "Solid",
    #             "lod": 2,
    #             "boundaries": [outer_shell]
    #         }]
    #     }
    #     cityjson['CityObjects']['room_'+str(i+1)] = building_part
    return False

def add_buildingpart_to_cjson(cityjson, parent_id, meshset, part_name):

    num_vertices = len(cityjson['vertices'])

    # Load building part
    meshset.meshing_merge_close_vertices()
    mesh = meshset.mesh(0)
    faces = mesh.face_matrix()
    vertices = mesh.vertex_matrix().astype(np.float16)

    # Create building part
    boundaries = (np.uint32(faces[np.newaxis,:,np.newaxis,:]+num_vertices)).tolist()
    building_part = {
        "type": "BuildingPart",
        "parents": [parent_id],
        "geometry": [{
            "type": "Solid",
            "lod": 2,
            "boundaries": boundaries
        }]
    }

    # Add building part & update vertices
    cityjson['CityObjects'][part_name] = building_part
    cityjson['vertices'].extend(vertices.tolist())

    return cityjson

def add_buildingroom_to_cjson(cityjson, parent_id, meshset, part_name):

    num_vertices = len(cityjson['vertices'])

    # Load building part
    meshset.meshing_merge_close_vertices()
    mesh = meshset.mesh(0)
    faces = mesh.face_matrix()
    vertices = mesh.vertex_matrix().astype(np.float16)

    # Create building part
    boundaries = (np.uint32(faces[np.newaxis,:,np.newaxis,:]+num_vertices)).tolist()
    building_part = {
        "type": "BuildingRoom",
        "parents": [parent_id],
        "geometry": [{
            "type": "Solid",
            "lod": 2,
            "boundaries": boundaries
        }]
    }

    # Add building part & update vertices
    cityjson['CityObjects'][part_name] = building_part
    cityjson['vertices'].extend(vertices.tolist())

    return cityjson

def save_to_file(cityjson, outfile):
    try:
        if os.path.isfile(outfile):
            os.remove(outfile)
        with open(outfile, "w", encoding="utf8") as file:
            json.dump(cityjson,file)
        logger.info(f'Succesfully saved output to {str(outfile)}')
    except:
        logger.error('Failed to save file.')

def to_cityjson_v1(rooms):
    """
    A function that converts a list of geometrical defined rooms to CityJSON v1.0 format.

    Parameters
    ----------
    rooms : list
        A list containing the rooms that should be converted. A room is a list of primitives/surfaces.
        Each primitive is a list of vertices (x,y,z), only linear and planar primitives are allowed.
        
    Returns
    -------
    A CityJSON object
    """

    # Assertions
    assert isinstance(rooms, list), "Argument rooms is not of type List."
    
    cityjson = {
        "type": "CityJSON",
        "version": "1.0",
        "CityObjects": {},
        "vertices": [],
        "transform": {
            "scale":[1.0,1.0,1.0], 
            "translate":[0.0,0.0,0.0]
        }
    }

    # add parent
    parent_id = "id-1"
    cityjson['CityObjects'][parent_id] = {"type":"Building", "geometry":[]}

    for i, meshset in enumerate(rooms):
        try:
            part_name = "room_" + str(i)
            add_buildingpart_to_cjson(cityjson, parent_id, meshset, part_name)
        except Exception as e:
            logger.error(f'Failed {part_name}')

    return cityjson

def to_cityjson_v1_1(rooms):
    """
    A function that converts a list of geometrical defined rooms to CityJSON v1.1.2 format.

    Parameters
    ----------
    rooms : list
        A list containing the rooms that should be converted. A room is a list of primitives/surfaces.
        Each primitive is a list of vertices (x,y,z), only linear and planar primitives are allowed.
    outfile : str
        The output path to be used for saving the CityJSON file.
        
    Returns
    -------
    A CityJSON object
    """

    # Assertions
    assert isinstance(rooms, list), "Argument rooms is not of type List."
    
    cityjson = {
        "type": "CityJSON",
        "version": "1.1",
        "CityObjects": {},
        "vertices": [],
        "transform": {
            "scale":[1.0,1.0,1.0], 
            "translate":[0.0,0.0,0.0]
        }
    }

    # add parent
    parent_id = "id-1"
    cityjson['CityObjects'][parent_id] = {"type":"Building"}

    # add parent
    parent_id = "id-1"
    cityjson['CityObjects'][parent_id] = {"type":"Building"}

    i = 1
    for meshset in rooms:
        try:
            part_name = "room_" + str(i)
            add_buildingroom_to_cjson(cityjson, parent_id, meshset, part_name)
        except Exception as e:
            logger.error(f'Failed {part_name}')
        i += 1

    return cityjson