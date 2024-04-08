from pathlib import Path

calibration_position = {
    # Val
    "scene_041": sorted([str(p) for p in Path("/workspace/videos/val/scene_041").glob("**/calibration.json")]),
    "scene_042": sorted([str(p) for p in Path("/workspace/videos/val/scene_042").glob("**/calibration.json")]),
    "scene_043": sorted([str(p) for p in Path("/workspace/videos/val/scene_043").glob("**/calibration.json")]),
    "scene_044": sorted([str(p) for p in Path("/workspace/videos/val/scene_044").glob("**/calibration.json")]),
    "scene_045": sorted([str(p) for p in Path("/workspace/videos/val/scene_045").glob("**/calibration.json")]),
    "scene_046": sorted([str(p) for p in Path("/workspace/videos/val/scene_046").glob("**/calibration.json")]),
    "scene_047": sorted([str(p) for p in Path("/workspace/videos/val/scene_047").glob("**/calibration.json")]),
    "scene_048": sorted([str(p) for p in Path("/workspace/videos/val/scene_048").glob("**/calibration.json")]),
    "scene_049": sorted([str(p) for p in Path("/workspace/videos/val/scene_049").glob("**/calibration.json")]),
    "scene_050": sorted([str(p) for p in Path("/workspace/videos/val/scene_050").glob("**/calibration.json")]),
    "scene_051": sorted([str(p) for p in Path("/workspace/videos/val/scene_051").glob("**/calibration.json")]),
    "scene_052": sorted([str(p) for p in Path("/workspace/videos/val/scene_052").glob("**/calibration.json")]),
    "scene_053": sorted([str(p) for p in Path("/workspace/videos/val/scene_053").glob("**/calibration.json")]),
    "scene_054": sorted([str(p) for p in Path("/workspace/videos/val/scene_054").glob("**/calibration.json")]),
    "scene_055": sorted([str(p) for p in Path("/workspace/videos/val/scene_055").glob("**/calibration.json")]),
    "scene_056": sorted([str(p) for p in Path("/workspace/videos/val/scene_056").glob("**/calibration.json")]),
    "scene_057": sorted([str(p) for p in Path("/workspace/videos/val/scene_057").glob("**/calibration.json")]),
    "scene_058": sorted([str(p) for p in Path("/workspace/videos/val/scene_058").glob("**/calibration.json")]),
    "scene_059": sorted([str(p) for p in Path("/workspace/videos/val/scene_059").glob("**/calibration.json")]),
    "scene_060": sorted([str(p) for p in Path("/workspace/videos/val/scene_060").glob("**/calibration.json")]),

    # Test
    "scene_061": sorted([str(p) for p in Path("/workspace/videos/test/scene_061").glob("**/calibration.json")]),
    "scene_062": sorted([str(p) for p in Path("/workspace/videos/test/scene_062").glob("**/calibration.json")]),
    "scene_063": sorted([str(p) for p in Path("/workspace/videos/test/scene_063").glob("**/calibration.json")]),
    "scene_064": sorted([str(p) for p in Path("/workspace/videos/test/scene_064").glob("**/calibration.json")]),
    "scene_065": sorted([str(p) for p in Path("/workspace/videos/test/scene_065").glob("**/calibration.json")]),
    "scene_066": sorted([str(p) for p in Path("/workspace/videos/test/scene_066").glob("**/calibration.json")]),
    "scene_067": sorted([str(p) for p in Path("/workspace/videos/test/scene_067").glob("**/calibration.json")]),
    "scene_068": sorted([str(p) for p in Path("/workspace/videos/test/scene_068").glob("**/calibration.json")]),
    "scene_069": sorted([str(p) for p in Path("/workspace/videos/test/scene_069").glob("**/calibration.json")]),
    "scene_070": sorted([str(p) for p in Path("/workspace/videos/test/scene_070").glob("**/calibration.json")]),
    "scene_071": sorted([str(p) for p in Path("/workspace/videos/test/scene_071").glob("**/calibration.json")]),
    "scene_072": sorted([str(p) for p in Path("/workspace/videos/test/scene_072").glob("**/calibration.json")]),
    "scene_073": sorted([str(p) for p in Path("/workspace/videos/test/scene_073").glob("**/calibration.json")]),
    "scene_074": sorted([str(p) for p in Path("/workspace/videos/test/scene_074").glob("**/calibration.json")]),
    "scene_075": sorted([str(p) for p in Path("/workspace/videos/test/scene_075").glob("**/calibration.json")]),
    "scene_076": sorted([str(p) for p in Path("/workspace/videos/test/scene_076").glob("**/calibration.json")]),
    "scene_077": sorted([str(p) for p in Path("/workspace/videos/test/scene_077").glob("**/calibration.json")]),
    "scene_078": sorted([str(p) for p in Path("/workspace/videos/test/scene_078").glob("**/calibration.json")]),
    "scene_079": sorted([str(p) for p in Path("/workspace/videos/test/scene_079").glob("**/calibration.json")]),
    "scene_080": sorted([str(p) for p in Path("/workspace/videos/test/scene_080").glob("**/calibration.json")]),
    "scene_081": sorted([str(p) for p in Path("/workspace/videos/test/scene_081").glob("**/calibration.json")]),
    "scene_082": sorted([str(p) for p in Path("/workspace/videos/test/scene_082").glob("**/calibration.json")]),
    "scene_083": sorted([str(p) for p in Path("/workspace/videos/test/scene_083").glob("**/calibration.json")]),
    "scene_084": sorted([str(p) for p in Path("/workspace/videos/test/scene_084").glob("**/calibration.json")]),
    "scene_085": sorted([str(p) for p in Path("/workspace/videos/test/scene_085").glob("**/calibration.json")]),
    "scene_086": sorted([str(p) for p in Path("/workspace/videos/test/scene_086").glob("**/calibration.json")]),
    "scene_087": sorted([str(p) for p in Path("/workspace/videos/test/scene_087").glob("**/calibration.json")]),
    "scene_088": sorted([str(p) for p in Path("/workspace/videos/test/scene_088").glob("**/calibration.json")]),
    "scene_089": sorted([str(p) for p in Path("/workspace/videos/test/scene_089").glob("**/calibration.json")]),
    "scene_090": sorted([str(p) for p in Path("/workspace/videos/test/scene_090").glob("**/calibration.json")]),
    


    "S005": [
        {
            "map_position": [(1536, 14), (1829, 1058), (373, 1057), (601, 13)],
            "cam_position": [(2, 844), (207, 382), (1613, 390), (1915, 865)],
            "name": "c025"
        },
        {
            "map_position": [(792, 976), (1583, 939), (1965, 0), (350, 0)],
            "cam_position": [(0, 1065), (1920, 1060), (1801, 468), (0,448)],
            "name": "c026"
        },
        {
            "map_position": [(259, 1026), (956, 1027), (1236, 0), (0, 0)],
            "cam_position": [(0, 1080), (1920, 1080), (1920, 408), (292, 409)],
            "name": "c027"
        },
        {
            "map_position": [(198, 817), (275, 1077), (1965, 1077), (991, 0)],
            "cam_position": [(0, 1080), (724, 1080), (1275, 266), (0, 238)],
            "name": "c028"
        },
        {
            "map_position": [(1913, 82), (1177, 24), (557, 1077), (1965, 1077)],
            "cam_position": [(0, 1080), (1920, 1080), (1920, 502), (440, 515)],
            "name": "c029"
        }
    ],
    "S008": [
        {
            "map_position": [(1496, 279), (1171, 214), (940, 473), (733, 1317), (1534, 1365)],
            "cam_position": [(0, 1080), (1920, 1080), (1885, 637), (1359, 388), (423, 405)],
            "name": "c041"
        },
        {
            "map_position": [(131, 1015), (451, 1059), (565, 897), (770, 63), (74, 16)],
            "cam_position": [(380, 1080), (1920, 1080), (1920, 805), (1446, 460), (556, 467)],
            "name": "c042"
        },
        {
            "map_position": [(433, 120), (212, 367), (508, 1316), (736, 1316), (1523, 1086), (1509, 350)],
            "cam_position": [(0, 1080), (1920, 1080), (1920, 356), (1548, 320), (782, 272), (0, 334)],
            "name": "c043"
        },
        {
            "map_position": [(1028, 1243), (1262, 950), (963, 180), (60, 250), (52, 1087)],
            "cam_position": [(0, 1080), (1920, 1080), (1920, 483), (954, 368), (20, 449)],
            "name": "c044"
        },
        {
            "map_position": [(163, 942), (372, 1217), (1523, 1054), (1509, 364), (534, 62)],
            "cam_position": [(0, 1080), (1920, 1080), (1920, 353), (1160, 288), (0, 386)],
            "name": "c045"
        },
        {
            "map_position": [(1216, 108), (71, 106), (52, 1043), (1036, 847), (1344, 366)],
            "cam_position": [(1458, 1080), (1729, 452), (743, 400), (0, 583), (0, 1080)],
            "name": "c046"
        }
    ],
    "S013": [
        {
            "map_position": [(573, 808), (761, 848), (932, 849), (928, 117), (579, 117), (576, 482), (415, 482), (63, 450), (67, 253), (70, 85)],
            "cam_position": [(99, 652), (672, 860), (1717, 1059), (1691, 321), (1061, 289), (759, 400), (447, 373), (43, 304), (244, 271), (436, 244)],
            "name": "c071"
        },
        {
            "map_position": [(723, 306), (1038, 433), (863, 1300), (55, 782)],
            "cam_position": [(1920, 1080), (127, 1080), (176, 475), (1721, 465)],
            "name": "c072"
        },
        {
            "map_position": [(347, 106), (167, 529), (245, 812), (1001, 734), (1039, 121), (575, 483), (416, 483)],
            "cam_position": [(26, 650), (0, 1080), (1920, 1080), (1894, 613), (1037, 519), (1033, 692), (739, 776)],
            "name": "c073"
        },
        {
            "map_position": [(585, 878), (742, 821), (813, 117), (69, 30), (62, 364)],
            "cam_position": [(234, 1080), (1920, 1080), (1920, 486), (569, 371), (165, 451)],
            "name": "c074"
        },
        {
            "map_position": [(309, 462), (288, 765), (931, 974), (1842, 926), (927, 400)],
            "cam_position": [(0, 1080), (1868, 1068), (1449, 529), (1046, 414), (425, 547)],
            "name": "c075"
        }
    ],
    "S017": [
        {
            "map_position": [(912, 697), (913, 387), (586, 219), (93, 215), (91, 738)],
            "cam_position": [(128, 1016), (1788, 1042), (1873, 587), (1450, 350), (724, 350)],
            "name": "c094"
        },
        {
            "map_position": [(913, 695), (1029, 476), (817, 168), (92, 223), (91, 738)],
            "cam_position": [(0, 640), (0, 1080), (1920, 1080), (1920, 348), (1228, 268)],
            "name": "c095"
        },
        {
            "map_position": [(133, 50), (145, 312), (502, 489), (605, 489), (810, 387), (758, 344), (810, 78), (758, 51)],
            "cam_position": [(0, 1080), (1920, 1080), (1894, 265), (1770, 169), (1380, 53), (1319, 80), (777, 48), (701, 78)],
            "name": "c096"
        },
        {
            "map_position": [(155, 387), (168, 681), (960, 702), (913, 387)],
            "cam_position": [(129, 1056), (1904, 1080), (1353, 27), (800, 32)],
            "name": "c097"
        },
        {
            "map_position": [(754, 681), (912, 591), (873, 101), (758, 50), (93, 48), (93, 683)],
            "cam_position": [(0, 798), (0, 1075), (1920, 1080), (1864, 824), (1380, 364), (568, 358)],
            "name": "c098"
        },
        {
            "map_position": [(217, 220), (141, 263), (134, 636), (207, 683), (556, 683), (584, 220)],
            "cam_position": [(0, 712), (0, 1080), (1920, 1080), (1920, 761), (1522, 0), (355, 0)],
            "name": "c099"
        }
    ],
    "S020": [
        {
            "map_position": [(771, 726), (410, 733), (335, 576), (361, 432), (361, 69), (681, 69), (682, 378), (812, 379)],
            "cam_position": [(1920, 1080), (213, 1012), (257, 674), (511, 534), (733, 355), (1319, 364), (1424, 514), (1918, 528)],
            "name": "c112"
        },
        {
            "map_position": [(843, 929), (350, 944), (120, 807), (126, 546), (358, 377), (359, 284), (849, 554)],
            "cam_position": [(0, 1080), (555, 459), (878, 378), (1306, 382), (1748, 472), (1920, 474), (1920, 1080)],
            "name": "c113"
        },
        {
            "map_position": [(714, 952), (120, 807), (127, 545), (336, 575), (362, 70), (682, 376), (928, 692)],
            "cam_position": [(42, 884), (287, 235), (746, 190), (878, 270), (1638, 183), (1789, 395), (1920, 1080)],
            "name": "c114"
        },
        {
            "map_position": [(959, 769), (713, 953), (349, 945), (120, 807), (124, 672), (702, 509)],
            "cam_position": [(76, 866), (596, 486), (1196, 365), (1636, 338), (1920, 377), (1891, 849)],
            "name": "c115"
        },
        {
            "map_position": [(684, 489), (1092, 506), (1098, 956), (713, 953), (547, 697)],
            "cam_position": [(516, 1080), (182, 558), (1109, 431), (1820, 615), (1920, 1080)],
            "name": "c116"
        },
        {
            "map_position": [(405, 642), (681, 70), (1088, 434), (993, 737), (681, 376), (1026, 386), (601, 856)],
            "cam_position": [(0, 1080), (216, 432), (1288, 420), (1831, 541), (552, 525), (1144, 427), (1920, 1080)],
            "name": "c117"
        }
    ],
    # Test
    "S001": [
        {
            "map_position": [(302, 535), (660, 502), (1478, 745), (740, 843), (372, 861)],
            "cam_position": [(165, 1035), (706, 342), (1138, 73), (1278, 282), (1531, 867)],
            "name": "c001"
        },
        {
            "map_position": [(1292, 642), (1290, 844), (771, 844), (252, 844), (146, 587)],
            "cam_position": [(1403, 961), (550, 945), (826, 410), (880, 305), (1040, 295)],
            "name": "c002"
        },
        {
            "map_position": [(146, 587), (373, 535), (373, 253), (302, 253), (222, 60)],
            "cam_position": [(752, 887), (1711, 714), (1354, 255), (1183, 252), (1037, 164)],
            "name": "c003"
        },
        {
            "map_position": [(373, 72), (373, 253), (519, 222), (660, 221), (730, 73), (739, 642)],
            "cam_position": [(515, 950), (1267, 949), (1046, 666), (1017, 529), (791, 487), (1745, 449)],
            "name": "c004"
        },
        {
            "map_position": [(373, 253), (252, 843), (740, 862), (739, 641)],
            "cam_position": [(357, 762), (925, 353), (416, 341), (265, 408)],
            "name": "c005"
        },
        {
            "map_position": [(302, 253), (660, 221), (739, 640), (519, 861), (372, 861)],
            "cam_position": [(1556, 754), (143, 848), (489, 287), (880, 225), (1051, 224)],
            "name": "c006"
        },
        {
            "map_position": [(145, 587), (151, 745), (740, 843), (738, 641), (519, 222), (407, 72), (145, 32)],
            "cam_position": [(94, 474), (65, 652), (1866, 787), (1529, 463), (915, 180), (784, 145), (507, 139)],
            "name": "c007"
        }
    ],
    "S003": [
        {
            "map_position": [(950, 742), (47, 764), (139, 10), (604, 10), (1323, 10)],
            "cam_position": [(1914, 776), (23, 798), (547, 424), (1078, 424), (1897, 423)],
            "name": "c014"
        },
        {
            "map_position": [(1324, 765), (532, 764), (125, 434), (528, 100), (942, 9), (1407, 10)],
            "cam_position": [(1891, 900), (128, 676), (6, 471), (696, 415), (1205, 424), (1893, 465)],
            "name": "c015"
        },
        {
            "map_position": [(1460, 23), (646, 10), (536, 752), (1148, 765), (1478, 784), (1535, 543)],
            "cam_position": [(60, 809), (1838, 823), (1529, 494), (860, 492), (507, 487), (343, 548)],
            "name": "c016"
        },
        {
            "map_position": [(1390, 784), (1325, 765), (926, 499), (815, 666), (588, 756), (512, 565), (25, 784), (5, 719)],
            "cam_position": [(1754, 1080), (1722, 960), (1887, 574), (1528, 514), (1315, 424), (1532, 400), (1183, 294), (1243, 291)],
            "name": "c017"
        },
        {
            "map_position": [(1238, 10), (1249, 434), (1082, 742), (534, 705), (12, 9)],
            "cam_position": [(693, 1007), (31, 809), (2, 645), (567, 504), (1432, 481)],
            "name": "c018"
        },
        {
            "map_position": [(267, 764), (1456, 764), (1759, 262), (1484, 0), (828, 138)],  
            "cam_position": [(8, 1025), (1403, 403), (1088, 314), (757, 321), (327, 440)],
            "name": "c019"
        }
    ],
    "S009": [
        {
            "map_position": [(1451, 289), (1172, 244), (940, 473), (942, 1255), (1244, 1320), (1501, 1320)],
            "cam_position": [(260, 1036), (1762, 997), (1889, 639), (1161, 404), (787, 405), (464, 411)],
            "name": "c047"
        },
        {
            "map_position": [(132, 989), (416, 1037), (986, 68), (463, 63), (96, 68)],
            "cam_position": [(413, 1010), (1734, 1042), (1702, 460), (1066, 466), (582, 474)],
            "name": "c048"
        },
        {
            "map_position": [(337, 288), (238, 564), (796, 1320), (1501, 1319), (1503, 613), (1494, 153)],
            "cam_position": [(275, 1041), (1781, 974), (1897, 334), (1304, 237), (634, 267), (39, 295)],
            "name": "c049"
        },
        {
            "map_position": [(1228, 1232), (1415, 943), (770, 64), (270, 63), (61, 419), (50, 1182)],
            "cam_position": [(72, 990), (1838, 1058), (1768, 278), (1319, 214), (903, 216), (59, 270)],
            "name": "c050"
        },
        {
            "map_position": [(327, 751), (273, 1085), (735, 1318), (1501, 1318), (1507, 753), (1501, 399)],
            "cam_position": [(115, 1068), (1843, 1037), (1618, 511), (1164, 338), (538, 348), (77, 355)],
            "name": "c051"
        },
        {
            "map_position": [(1269, 705), (1312, 381), (985, 69), (72, 68), (52, 957)],
            "cam_position": [(69, 1080), (1846, 959), (1884, 471), (1112, 244), (69, 279)],
            "name": "c052"
        }
    ],
    "S014": [
        {
            "map_position": [(576, 481), (58, 807), (58, 940), (242, 807), (575, 807), (761, 847), (875, 1302), (1001, 550)], 
            "cam_position": [(1697, 711), (1672, 503), (1496, 483), (1494, 525), (1071, 578), (671, 608), (302, 509), (295, 955)],  
            "name": "c076"
        },
        {
            "map_position": [(416, 483), (247, 763), (932, 923), (1848, 918), (1000, 734), (928, 495)],
            "cam_position": [(255, 749), (1840, 1013), (1295, 344), (990, 211), (949, 333), (519, 365)],
            "name": "c077"
        },
        {
            "map_position": [(61, 674), (59, 873), (573, 808), (571, 1300), (743, 1301), (761, 847)],
            "cam_position": [(1859, 360), (1499, 323), (1055, 516), (344, 353), (0, 403), (546, 614)],
            "name": "c078"
        },
        {
            "map_position": [(54, 1071), (70, 32), (419, 32), (244, 945)],
            "cam_position": [(581, 906), (689, 387), (1112, 380), (1306, 685)],
            "name": "c079"
        },
        {
            "map_position": [(574, 809), (932, 923), (1567, 925), (1001, 735), (977, 696)],
            "cam_position": [(1671, 988), (808, 459), (159, 268), (193, 521), (87, 565)],
            "name": "c080"
        },
        {
            "map_position": [(934, 119), (569, 117), (575, 482), (416, 482), (67, 185)],
            "cam_position": [(1722, 445), (1065, 407), (760, 541), (427, 511), (287, 371)],
            "name": "c081"
        }
    ],
    "S018": [
        {
            "map_position": [(88, 158), (86, 590), (172, 678), (871, 682), (867, 420), (862, 77)],
            "cam_position": [(0, 897), (1728, 898), (1895, 731), (1372, 245), (990, 245), (487, 244)],
            "name": "c100"
        },
        {
            "map_position": [(260, 504), (171, 855), (964, 963), (953, 508)],
            "cam_position": [(133, 918), (1768, 926), (1093, 312), (393, 342)],
            "name": "c101"
        },
        {
            "map_position": [(9, 42), (0, 590), (348, 680), (304, 46), (347, 245), (304, 472)],
            "cam_position": [(528, 168), (55, 705), (1734, 907), (1096, 162), (1299, 326), (1270, 480)],
            "name": "c102"
        },
        {
            "map_position": [(678, 770), (783, 682), (782, 515), (87, 503), (85, 769)],
            "cam_position": [(229, 718), (576, 1062), (1704, 1025), (1105, 164), (614, 169)],
            "name": "c103"
        },
        {
            "map_position": [(782, 595), (779, 334), (692, 247), (304, 84), (3, 157), (0, 589)],
            "cam_position": [(331, 905), (1679, 851), (1851, 654), (1826, 344), (1407, 219), (668, 229)],
            "name": "c104"
        },
        {
            "map_position": [(2, 329), (347, 331), (434, 419), (437, 960), (171, 956)],
            "cam_position": [(1920, 836), (194, 810), (12, 618), (548, 192), (1185, 196)],
            "name": "c105"
        }
    ],
    "S021": [
        {
            "map_position": [(1185, 165), (1290, 324), (1295, 732), (1054, 731), (629, 542), (1010, 321)],
            "cam_position": [(1677, 822), (914, 820), (0, 426), (487, 240), (1234, 87), (1346, 409)],
            "name": "c118"
        },
        {
            "map_position": [(1407, 141), (1290, 544), (1009, 542), (805, 321), (1009, 320), (1057, 139)],
            "cam_position": [(1875, 392), (789, 837), (175, 448), (403, 153), (688, 253), (1065, 163)],
            "name": "c119"
        },
        {
            "map_position": [(807, 157), (855, 340), (854, 560), (1057, 339), (1602, 199), (525, 140), (520, 536)],
            "cam_position": [(575, 494), (969, 547), (1494, 674), (1209, 451), (1394, 281), (43, 650), (962, 1080)],
            "name": "c120"
        },
        {
            "map_position": [(855, 339), (854, 560), (806, 748), (1055, 558), (1612, 711), (516, 731), (522, 324)],
            "cam_position": [(220, 605), (760, 472), (1142, 414), (511, 392), (326, 233), (1630, 530), (747, 956)],
            "name": "c121"
        },
        {
            "map_position": [(342, 748), (806, 749), (854, 560), (533, 324)],
            "cam_position": [(1225, 255), (198, 494), (115, 858), (1484, 1011)],
            "name": "c122"
        },
        {
            "map_position": [(1055, 731), (1002, 542), (808, 543), (255, 539), (262, 320), (806, 323), (1009, 321), (1056, 140)],
            "cam_position": [(70, 796), (658, 736), (744, 526), (857, 249), (1145, 252), (1223, 532), (1288, 752), (1904, 858)],
            "name": "c123"
        }
    ],
    "S022": [
        {
            "map_position": [(1185, 165), (1290, 324), (1295, 732), (1054, 731), (629, 542), (1010, 321)],
            "cam_position": [(1677, 822), (914, 820), (0, 426), (487, 240), (1234, 87), (1346, 409)],
            "name": "c124"
        },
        {
            "map_position": [(1407, 141), (1290, 544), (1009, 542), (805, 321), (1009, 320), (1057, 139)],
            "cam_position": [(1875, 392), (789, 837), (175, 448), (403, 153), (688, 253), (1065, 163)],
            "name": "c125"
        },
        {
            "map_position": [(807, 157), (855, 340), (854, 560), (1057, 339), (1602, 199), (525, 140), (520, 536)],
            "cam_position": [(575, 494), (969, 547), (1494, 674), (1209, 451), (1394, 281), (43, 650), (962, 1080)],
            "name": "c126"
        },
        {
            "map_position": [(855, 339), (854, 560), (806, 748), (1055, 558), (1612, 711), (516, 731), (522, 324)],
            "cam_position": [(220, 605), (760, 472), (1142, 414), (511, 392), (326, 233), (1630, 530), (747, 956)],
            "name": "c127"
        },
        {
            "map_position": [(342, 748), (806, 749), (854, 560), (533, 324)],
            "cam_position": [(1225, 255), (198, 494), (115, 858), (1484, 1011)],
            "name": "c128"
        },
        {
            "map_position": [(1110, 155), (1062, 337), (1061, 558), (860, 339), (318, 196), (316, 260)],
            "cam_position": [(1837, 634), (1282, 621), (600, 697), (1037, 510), (929, 351), (846, 355)],
            "name": "c129"
        }
    ],
}
