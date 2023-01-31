#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: ored
# date: 2020/9/16
# update: 2020/9/16
"""
get_material_from_item_center.py:
"""

from istio_pyservice_utils import itemcenter_client


def get_item_info(item_id):
    materials = itemcenter_client.get_item_info(item_id=item_id)
    item_info = {'item_id': str(item_id),
                 'title': materials.title
                 }
    # if materials.videoUrl is not None:
    #     item_info['videos'].append(materials.videoUrl)
    # if materials.pictUrl is not None:
    #     item_info['imgs'].append(
    #         {
    #             'tfs': materials.pictUrl,
    #             'img_pos': 0,
    #             'is_wl': 0
    #         }
    #     )
    # for idx, tfs in enumerate(materials.subImageList):
    #     item_info['imgs'].append(
    #         {
    #             'tfs': tfs,
    #             'img_pos': idx + 1,
    #             'is_wl': 0
    #         }
    #     )
    # for idx, tfs in enumerate(materials.creativeImageList):
    #     item_info['imgs'].append(
    #         {
    #             'tfs': tfs,
    #             'img_pos': -1,
    #             'is_wl': 0
    #         }
    #     )
    # for idx, tfs in enumerate(materials.skuImageList):
    #     item_info['imgs'].append(
    #         {
    #             'tfs': tfs,
    #             'img_pos': -2,
    #             'is_wl': 0
    #         }
    #     )
    if len(materials.category_list) > 0:
        item_info['cate_id'] = materials.category_list[0]['id']
        item_info['cate_name'] = materials.category_list[0]['name']
        item_info['cate_level1_id'] = materials.category_list[-1]['id']
        item_info['cate_level1_name'] = materials.category_list[-1]['name']
    if len(materials.category_list) > 1:
        item_info['cate_level2_id'] = materials.category_list[-2]['id']
        item_info['cate_level2_name'] = materials.category_list[-2]['name']
    return item_info


if __name__ == '__main__':
    item_info = get_item_info('603269207582')
    print(item_info)
