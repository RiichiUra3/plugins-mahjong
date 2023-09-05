import json

from mahjong_utils.models.tile import parse_tiles
from mahjong_utils.shanten import shanten

from maybee.network import MahjongPlayer
from maybee.arena.common import human_to_tile

from copy import deepcopy

import torch
import numpy as np


structured_human_to_tile = deepcopy(human_to_tile)
structured_human_to_tile["1z"] = human_to_tile["dong"]
structured_human_to_tile["2z"] = human_to_tile["nan"]
structured_human_to_tile["3z"] = human_to_tile["xi"]
structured_human_to_tile["4z"] = human_to_tile["bei"]
structured_human_to_tile["5z"] = human_to_tile["bai"]
structured_human_to_tile["6z"] = human_to_tile["fa"]
structured_human_to_tile["7z"] = human_to_tile["zhong"]

ai_player = MahjongPlayer()
ai_player.load_state_dict(torch.load("checkpoint/resume.pth")["best_network_params"])
ai_player.to(0)
ai_player.eval()

RECORD_PAD = torch.zeros(1, 55)

def construct_input(tiles):
    self_info = torch.zeros(34, 18, dtype=torch.float32)
    for tile in tiles:
        pos = structured_human_to_tile[tile]
        num = 0
        while self_info[pos, num] != 0:
            num += 1
        self_info[pos, num] = 1
        if tile in ["0m", "0p", "0s"]:
            self_info[pos, 6]
            
    # set last tile to tsumo tile
    pos = structured_human_to_tile[tiles[-1]]
    self_info[pos, 9] = 1
    
    # set game_wind and self_wind to dong
    pos = structured_human_to_tile["1z"]
    self_info[pos, 7] = self_info[pos, 8] = 1
    
    # set dora indicator to xi
    pos = structured_human_to_tile["3z"]
    self_info[pos, 5] = 1
    
    # set dora to bei
    pos = structured_human_to_tile["4z"]
    self_info[pos, 4] = 1
    
    record_info = RECORD_PAD
    global_info = torch.Tensor([0, 7, 0, 0, 0, 0, 250, 250, 250, 250, 0, 0, 0, 0, 69]).to(torch.float32)
    valid_actions_mask = np.zeros(47, dtype=np.float32)
    for tile in tiles:
        pos = structured_human_to_tile[tile]
        valid_actions_mask[pos] = 1
    return self_info, record_info, global_info, valid_actions_mask
    
    
def get_ai_pred(hand):
    tiles = [str(t) for t in parse_tiles(hand)]
    self_info, record_info, global_info, valid_actions_mask = construct_input(tiles)
    self_info = self_info.unsqueeze(0).to(0)
    record_info = record_info.unsqueeze(0).to(0)
    global_info = global_info.unsqueeze(0).to(0)
    pred = ai_player(self_info, record_info, global_info)
    pred = torch.softmax(pred[0], dim=-1).cpu().detach().numpy()
    masked_pred = pred * valid_actions_mask
    masked_pred = masked_pred / masked_pred.sum()
    ret = {}
    ret["AI_recommendation"] = {}
    for tile in tiles:
        ret["AI_recommendation"][tile] = masked_pred[structured_human_to_tile[tile]]
    return ret


def get_paili_json(hand):
    print(hand)
    result = shanten(parse_tiles(hand))
    ai_pred = get_ai_pred(hand)
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
        for section in ai_pred:
            result_array[-1][section] = float(ai_pred[section][str(tile)])
    return json.dumps(result_array)