import json

from mahjong_utils.models.tile import parse_tiles
from mahjong_utils.shanten import shanten

def get_paili_json(hand):
    print(hand)
    result = shanten(parse_tiles(hand))
    result_array = []
    for tile in result.discard_to_advance:
        result_discard_tile = result.discard_to_advance[tile]
        result_array.append({
            "discard": str(tile),
            "shanten": result_discard_tile.shanten,
            "advance": [] if result_discard_tile.advance is None else [str(t) for t in result_discard_tile.advance],
            "advance_num": result_discard_tile.advance_num or 0,
            "good_shape_advance": [] if result_discard_tile.good_shape_advance is None else [str(t) for t in result_discard_tile.good_shape_advance],
            "good_shape_advance_num": result_discard_tile.good_shape_advance_num or 0
        })
    return json.dumps(result_array)