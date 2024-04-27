
noise = [0.3741, 0.5417, 0.7552, 0.7973, 0.7908, 0.7392, 0.7219, 0.7087, 0.6808,
        0.6472, 0.6215, 0.6107, 0.5729, 0.5517, 0.5393, 0.5028, 0.4972, 0.4932,
        0.4592, 0.4849, 0.4467, 0.4685, 0.4447, 0.4299, 0.4231, 0.4464, 0.3988,
        0.3910, 0.4093, 0.3671, 0.3742, 0.3559, 0.3713, 0.3513, 0.3473, 0.3253,
        0.3174, 0.3182, 0.3320, 0.2962, 0.3011, 0.2843, 0.3078, 0.2828, 0.2761,
        0.2861, 0.2634, 0.2668, 0.2577, 0.2488, 0.2728, 0.2536, 0.2378, 0.2427,
        0.2315, 0.2266, 0.2431, 0.2371, 0.2177, 0.2186, 0.2234, 0.2120, 0.2093,
        0.2199, 0.2115, 0.1999, 0.2024, 0.2117, 0.1999, 0.1930, 0.1925, 0.1946,
        0.1845, 0.1804, 0.1843, 0.1938, 0.1887, 0.1777, 0.1726, 0.1735, 0.1736,
        0.1652, 0.1624, 0.1637, 0.1700, 0.1677, 0.1580, 0.1541, 0.1544, 0.1556,
        0.1544, 0.1491, 0.1470, 0.1463, 0.1480, 0.1489, 0.1432, 0.1387, 0.1385,
        0.1403, 0.1428, 0.1414, 0.1355, 0.1333, 0.1321, 0.1318, 0.1324, 0.1305,
        0.1264, 0.1252, 0.1234, 0.1240, 0.1261, 0.1274, 0.1219, 0.1178, 0.1166,
        0.1158, 0.1153, 0.1153, 0.1142, 0.1114, 0.1100, 0.1087, 0.1077, 0.1072,
        0.1086, 0.1078, 0.1051, 0.1026, 0.1019, 0.1021, 0.1018, 0.1017, 0.1023,
        0.1001, 0.0981, 0.0976, 0.0970, 0.0954, 0.0952, 0.0956, 0.0953, 0.0936,
        0.0918, 0.0909, 0.0903, 0.0899, 0.0897, 0.0901, 0.0915, 0.0917, 0.0896,
        0.0879, 0.0868, 0.0861, 0.0853, 0.0848, 0.0846, 0.0842, 0.0838, 0.0827,
        0.0818, 0.0809, 0.0805, 0.0799, 0.0798, 0.0797, 0.0797, 0.0800, 0.0784,
        0.0773, 0.0765, 0.0759, 0.0752, 0.0747, 0.0741, 0.0733, 0.0730, 0.0734,
        0.0724, 0.0719, 0.0712, 0.0710, 0.0709, 0.0702, 0.0698, 0.0694, 0.0697,
        0.0696, 0.0694, 0.0686, 0.0682, 0.0671, 0.0669, 0.0667, 0.0667, 0.0665,
        0.0664, 0.0663, 0.0663, 0.0668, 0.0658, 0.0651, 0.0647, 0.0642, 0.0638,
        0.0636, 0.0633, 0.0632, 0.0632, 0.0629, 0.0627, 0.0626, 0.0617, 0.0614,
        0.0610, 0.0606, 0.0603, 0.0601, 0.0594, 0.0589, 0.0586, 0.0583, 0.0576,
        0.0566, 0.0553, 0.0537, 0.0515, 0.0492, 0.0467, 0.0438, 0.0409, 0.0377,
        0.0343, 0.0310, 0.0279, 0.0249, 0.0222, 0.0197, 0.0177, 0.0159, 0.0146,
        0.0135, 0.0128, 0.0123, 0.0119, 0.0117, 0.0115, 0.0115, 0.0114, 0.0113,
        0.0113, 0.0112, 0.0112, 0.0112, 0.0112]
noisy = [1.0900, 1.0602, 1.4281, 2.3177, 2.8135, 2.6463, 2.7954, 2.7798, 2.4669,
        2.2285, 2.1283, 2.1105, 2.0554, 2.0178, 1.9683, 1.7966, 1.6397, 1.5095,
        1.3805, 1.2964, 1.1785, 1.1268, 1.0270, 0.9572, 0.9019, 0.8732, 0.7935,
        0.7552, 0.7438, 0.6851, 0.6709, 0.6370, 0.6390, 0.6093, 0.5992, 0.5716,
        0.5580, 0.5511, 0.5569, 0.5195, 0.5179, 0.4972, 0.5174, 0.4906, 0.4827,
        0.4907, 0.4669, 0.4678, 0.4572, 0.4446, 0.4629, 0.4416, 0.4229, 0.4221,
        0.4069, 0.3970, 0.4069, 0.3960, 0.3728, 0.3678, 0.3674, 0.3524, 0.3450,
        0.3510, 0.3399, 0.3263, 0.3259, 0.3328, 0.3201, 0.3130, 0.3125, 0.3149,
        0.3057, 0.3030, 0.3072, 0.3168, 0.3133, 0.3037, 0.2994, 0.3004, 0.2997,
        0.2908, 0.2865, 0.2856, 0.2889, 0.2841, 0.2726, 0.2662, 0.2643, 0.2634,
        0.2600, 0.2530, 0.2491, 0.2466, 0.2467, 0.2463, 0.2398, 0.2346, 0.2335,
        0.2341, 0.2356, 0.2335, 0.2272, 0.2246, 0.2231, 0.2224, 0.2228, 0.2209,
        0.2167, 0.2152, 0.2129, 0.2128, 0.2141, 0.2146, 0.2088, 0.2044, 0.2024,
        0.2011, 0.1999, 0.1990, 0.1968, 0.1933, 0.1914, 0.1896, 0.1885, 0.1880,
        0.1892, 0.1878, 0.1836, 0.1796, 0.1777, 0.1771, 0.1760, 0.1754, 0.1753,
        0.1726, 0.1704, 0.1695, 0.1684, 0.1664, 0.1659, 0.1661, 0.1655, 0.1637,
        0.1618, 0.1608, 0.1600, 0.1593, 0.1587, 0.1589, 0.1601, 0.1601, 0.1581,
        0.1564, 0.1551, 0.1540, 0.1531, 0.1524, 0.1521, 0.1513, 0.1508, 0.1497,
        0.1488, 0.1477, 0.1473, 0.1467, 0.1464, 0.1462, 0.1462, 0.1462, 0.1445,
        0.1434, 0.1425, 0.1419, 0.1411, 0.1405, 0.1399, 0.1388, 0.1385, 0.1388,
        0.1379, 0.1373, 0.1364, 0.1361, 0.1358, 0.1350, 0.1344, 0.1338, 0.1338,
        0.1335, 0.1331, 0.1321, 0.1315, 0.1302, 0.1296, 0.1292, 0.1289, 0.1286,
        0.1283, 0.1282, 0.1281, 0.1285, 0.1276, 0.1268, 0.1264, 0.1258, 0.1253,
        0.1251, 0.1247, 0.1244, 0.1243, 0.1238, 0.1236, 0.1232, 0.1223, 0.1219,
        0.1213, 0.1208, 0.1200, 0.1194, 0.1182, 0.1174, 0.1164, 0.1154, 0.1138,
        0.1117, 0.1092, 0.1060, 0.1022, 0.0978, 0.0930, 0.0876, 0.0820, 0.0761,
        0.0698, 0.0637, 0.0579, 0.0524, 0.0474, 0.0429, 0.0391, 0.0360, 0.0336,
        0.0317, 0.0303, 0.0294, 0.0288, 0.0284, 0.0282, 0.0280, 0.0279, 0.0278,
        0.0277, 0.0276, 0.0276, 0.0275, 0.0275]
clean = [0.8367, 0.6600, 0.8598, 1.7670, 2.3002, 2.1797, 2.3497, 2.3408, 2.0383,
        1.8179, 1.7342, 1.7217, 1.6937, 1.6684, 1.6234, 1.4748, 1.3128, 1.1767,
        1.0691, 0.9546, 0.8632, 0.7860, 0.7011, 0.6398, 0.5863, 0.5318, 0.4918,
        0.4573, 0.4258, 0.4037, 0.3804, 0.3613, 0.3465, 0.3338, 0.3260, 0.3178,
        0.3101, 0.3013, 0.2927, 0.2877, 0.2807, 0.2748, 0.2728, 0.2688, 0.2670,
        0.2652, 0.2619, 0.2596, 0.2570, 0.2522, 0.2475, 0.2435, 0.2390, 0.2330,
        0.2273, 0.2214, 0.2152, 0.2090, 0.2032, 0.1966, 0.1909, 0.1857, 0.1803,
        0.1759, 0.1721, 0.1687, 0.1656, 0.1633, 0.1612, 0.1604, 0.1603, 0.1607,
        0.1608, 0.1618, 0.1623, 0.1629, 0.1640, 0.1645, 0.1650, 0.1651, 0.1640,
        0.1627, 0.1606, 0.1584, 0.1556, 0.1526, 0.1496, 0.1463, 0.1438, 0.1414,
        0.1388, 0.1362, 0.1338, 0.1316, 0.1300, 0.1284, 0.1269, 0.1257, 0.1246,
        0.1234, 0.1223, 0.1213, 0.1203, 0.1195, 0.1189, 0.1184, 0.1180, 0.1177,
        0.1172, 0.1167, 0.1160, 0.1153, 0.1144, 0.1136, 0.1128, 0.1121, 0.1109,
        0.1101, 0.1092, 0.1082, 0.1069, 0.1057, 0.1049, 0.1042, 0.1040, 0.1038,
        0.1036, 0.1029, 0.1010, 0.0990, 0.0975, 0.0964, 0.0955, 0.0949, 0.0940,
        0.0933, 0.0928, 0.0921, 0.0914, 0.0909, 0.0904, 0.0901, 0.0897, 0.0893,
        0.0890, 0.0886, 0.0883, 0.0879, 0.0873, 0.0871, 0.0868, 0.0866, 0.0864,
        0.0862, 0.0858, 0.0853, 0.0850, 0.0847, 0.0844, 0.0841, 0.0838, 0.0836,
        0.0834, 0.0831, 0.0830, 0.0829, 0.0827, 0.0824, 0.0823, 0.0821, 0.0817,
        0.0815, 0.0814, 0.0812, 0.0810, 0.0809, 0.0808, 0.0803, 0.0802, 0.0802,
        0.0802, 0.0800, 0.0797, 0.0794, 0.0792, 0.0789, 0.0787, 0.0785, 0.0782,
        0.0778, 0.0776, 0.0773, 0.0770, 0.0766, 0.0762, 0.0760, 0.0756, 0.0755,
        0.0753, 0.0753, 0.0751, 0.0750, 0.0750, 0.0749, 0.0748, 0.0746, 0.0745,
        0.0744, 0.0742, 0.0741, 0.0739, 0.0737, 0.0736, 0.0733, 0.0731, 0.0729,
        0.0727, 0.0725, 0.0720, 0.0715, 0.0710, 0.0704, 0.0697, 0.0688, 0.0678,
        0.0666, 0.0651, 0.0634, 0.0614, 0.0590, 0.0564, 0.0535, 0.0503, 0.0471,
        0.0438, 0.0405, 0.0374, 0.0344, 0.0317, 0.0293, 0.0273, 0.0256, 0.0242,
        0.0232, 0.0225, 0.0219, 0.0216, 0.0214, 0.0212, 0.0211, 0.0210, 0.0210,
        0.0209, 0.0209, 0.0209, 0.0209, 0.0208]
ssl_noise = [0.1523, 0.2127, 0.3435, 0.4476, 0.4915, 0.4522, 0.4515, 0.5268, 0.5727,
        0.5324, 0.4642, 0.4532, 0.4504, 0.4294, 0.3897, 0.3427, 0.3299, 0.3162,
        0.2819, 0.2818, 0.2438, 0.2421, 0.2258, 0.2214, 0.2364, 0.2819, 0.2729,
        0.2840, 0.3247, 0.3085, 0.3140, 0.2905, 0.3145, 0.3350, 0.3661, 0.3688,
        0.3867, 0.4353, 0.5081, 0.4908, 0.5181, 0.4917, 0.5509, 0.5395, 0.5662,
        0.5950, 0.5187, 0.4998, 0.4830, 0.4727, 0.5038, 0.4411, 0.3888, 0.3785,
        0.3422, 0.3131, 0.3085, 0.2725, 0.2273, 0.2098, 0.1961, 0.1686, 0.1502,
        0.1424, 0.1256, 0.1093, 0.1005, 0.0943, 0.0825, 0.0752, 0.0741, 0.0776,
        0.0814, 0.0892, 0.1023, 0.1218, 0.1349, 0.1454, 0.1602, 0.1811, 0.2014,
        0.2124, 0.2286, 0.2465, 0.2637, 0.2583, 0.2335, 0.2156, 0.2055, 0.1972,
        0.1870, 0.1745, 0.1692, 0.1656, 0.1650, 0.1638, 0.1552, 0.1458, 0.1395,
        0.1354, 0.1305, 0.1206, 0.1090, 0.1077, 0.1137, 0.1193, 0.1190, 0.1121,
        0.1076, 0.1138, 0.1203, 0.1219, 0.1179, 0.1143, 0.1150, 0.1214, 0.1239,
        0.1167, 0.1058, 0.0993, 0.0951, 0.0875, 0.0784, 0.0710, 0.0687, 0.0732,
        0.0836, 0.0932, 0.0995, 0.1060, 0.1164, 0.1295, 0.1394, 0.1452, 0.1492,
        0.1491, 0.1478, 0.1447, 0.1394, 0.1330, 0.1285, 0.1230, 0.1150, 0.1054,
        0.0967, 0.0896, 0.0834, 0.0786, 0.0747, 0.0701, 0.0651, 0.0599, 0.0550,
        0.0505, 0.0451, 0.0396, 0.0354, 0.0321, 0.0280, 0.0232, 0.0191, 0.0172,
        0.0173, 0.0193, 0.0228, 0.0266, 0.0301, 0.0335, 0.0378, 0.0431, 0.0473,
        0.0506, 0.0530, 0.0559, 0.0591, 0.0619, 0.0633, 0.0642, 0.0663, 0.0691,
        0.0696, 0.0684, 0.0661, 0.0645, 0.0641, 0.0633, 0.0617, 0.0601, 0.0604,
        0.0611, 0.0602, 0.0577, 0.0558, 0.0544, 0.0534, 0.0511, 0.0479, 0.0449,
        0.0430, 0.0411, 0.0382, 0.0351, 0.0321, 0.0300, 0.0281, 0.0262, 0.0249,
        0.0244, 0.0239, 0.0232, 0.0226, 0.0222, 0.0220, 0.0218, 0.0213, 0.0211,
        0.0211, 0.0212, 0.0213, 0.0213, 0.0213, 0.0215, 0.0217, 0.0215, 0.0209,
        0.0199, 0.0189, 0.0179, 0.0169, 0.0157, 0.0144, 0.0131, 0.0119, 0.0110,
        0.0104, 0.0100, 0.0098, 0.0096, 0.0095, 0.0094, 0.0094, 0.0093, 0.0093,
        0.0093, 0.0092, 0.0092, 0.0092, 0.0092, 0.0092, 0.0091, 0.0091, 0.0091,
        0.0091, 0.0091, 0.0091, 0.0091, 0.0091]
ssl_noisy = [0.5082, 0.5430, 0.7964, 1.4848, 1.9672, 1.8505, 2.0382, 2.4006, 2.4046,
        2.1424, 1.8830, 1.8637, 1.9223, 1.8751, 1.7080, 1.4747, 1.3042, 1.1564,
        1.0144, 0.8990, 0.7707, 0.6968, 0.6222, 0.5848, 0.5914, 0.6337, 0.6246,
        0.6287, 0.6659, 0.6533, 0.6366, 0.5895, 0.6106, 0.6522, 0.7091, 0.7284,
        0.7646, 0.8436, 0.9520, 0.9699, 1.0002, 0.9706, 1.0378, 1.0591, 1.1220,
        1.1545, 1.0481, 0.9964, 0.9778, 0.9652, 0.9683, 0.8726, 0.7894, 0.7489,
        0.6861, 0.6270, 0.5862, 0.5165, 0.4444, 0.4031, 0.3686, 0.3223, 0.2861,
        0.2633, 0.2352, 0.2096, 0.1916, 0.1772, 0.1593, 0.1481, 0.1459, 0.1515,
        0.1607, 0.1763, 0.1990, 0.2305, 0.2591, 0.2870, 0.3205, 0.3608, 0.4001,
        0.4311, 0.4649, 0.4941, 0.5121, 0.4989, 0.4607, 0.4260, 0.4010, 0.3798,
        0.3577, 0.3370, 0.3259, 0.3171, 0.3116, 0.3069, 0.2949, 0.2802, 0.2673,
        0.2564, 0.2444, 0.2263, 0.2083, 0.2069, 0.2187, 0.2291, 0.2279, 0.2163,
        0.2108, 0.2231, 0.2367, 0.2387, 0.2279, 0.2192, 0.2247, 0.2404, 0.2452,
        0.2311, 0.2095, 0.1960, 0.1875, 0.1739, 0.1567, 0.1426, 0.1387, 0.1478,
        0.1672, 0.1857, 0.1985, 0.2109, 0.2300, 0.2537, 0.2721, 0.2822, 0.2877,
        0.2896, 0.2891, 0.2830, 0.2722, 0.2611, 0.2520, 0.2408, 0.2253, 0.2085,
        0.1932, 0.1798, 0.1680, 0.1589, 0.1508, 0.1417, 0.1310, 0.1208, 0.1126,
        0.1045, 0.0944, 0.0837, 0.0757, 0.0694, 0.0616, 0.0522, 0.0444, 0.0407,
        0.0410, 0.0450, 0.0520, 0.0598, 0.0668, 0.0736, 0.0821, 0.0925, 0.1016,
        0.1089, 0.1145, 0.1207, 0.1277, 0.1338, 0.1374, 0.1395, 0.1441, 0.1498,
        0.1518, 0.1498, 0.1453, 0.1420, 0.1410, 0.1396, 0.1364, 0.1333, 0.1333,
        0.1343, 0.1325, 0.1278, 0.1238, 0.1213, 0.1190, 0.1140, 0.1069, 0.1005,
        0.0965, 0.0924, 0.0861, 0.0792, 0.0732, 0.0690, 0.0650, 0.0611, 0.0584,
        0.0573, 0.0563, 0.0549, 0.0535, 0.0526, 0.0522, 0.0517, 0.0510, 0.0506,
        0.0506, 0.0509, 0.0510, 0.0510, 0.0510, 0.0516, 0.0517, 0.0512, 0.0497,
        0.0476, 0.0453, 0.0432, 0.0410, 0.0385, 0.0357, 0.0329, 0.0303, 0.0283,
        0.0270, 0.0261, 0.0256, 0.0252, 0.0250, 0.0249, 0.0247, 0.0246, 0.0246,
        0.0245, 0.0244, 0.0244, 0.0243, 0.0243, 0.0243, 0.0242, 0.0242, 0.0242,
        0.0242, 0.0241, 0.0241, 0.0241, 0.0241]
ssl_clean = [0.4144, 0.3985, 0.5498, 1.1804, 1.6500, 1.5709, 1.7679, 2.0795, 2.0475,
        1.8111, 1.5986, 1.5858, 1.6467, 1.6119, 1.4688, 1.2651, 1.0969, 0.9527,
        0.8327, 0.7105, 0.6089, 0.5318, 0.4674, 0.4318, 0.4249, 0.4272, 0.4265,
        0.4206, 0.4213, 0.4240, 0.4004, 0.3722, 0.3710, 0.3966, 0.4280, 0.4467,
        0.4686, 0.5071, 0.5529, 0.5906, 0.5967, 0.5911, 0.6059, 0.6417, 0.6851,
        0.6915, 0.6510, 0.6131, 0.6090, 0.6059, 0.5777, 0.5348, 0.4952, 0.4605,
        0.4270, 0.3907, 0.3499, 0.3083, 0.2737, 0.2453, 0.2203, 0.1963, 0.1744,
        0.1566, 0.1422, 0.1297, 0.1184, 0.1083, 0.0999, 0.0946, 0.0932, 0.0960,
        0.1024, 0.1119, 0.1240, 0.1394, 0.1580, 0.1783, 0.2008, 0.2246, 0.2478,
        0.2713, 0.2927, 0.3075, 0.3104, 0.3011, 0.2835, 0.2628, 0.2451, 0.2296,
        0.2151, 0.2044, 0.1972, 0.1909, 0.1854, 0.1811, 0.1763, 0.1693, 0.1611,
        0.1531, 0.1446, 0.1343, 0.1259, 0.1255, 0.1324, 0.1383, 0.1371, 0.1311,
        0.1294, 0.1367, 0.1452, 0.1458, 0.1379, 0.1318, 0.1371, 0.1479, 0.1505,
        0.1421, 0.1291, 0.1207, 0.1154, 0.1079, 0.0980, 0.0898, 0.0877, 0.0932,
        0.1041, 0.1150, 0.1227, 0.1298, 0.1403, 0.1530, 0.1634, 0.1686, 0.1705,
        0.1725, 0.1731, 0.1692, 0.1626, 0.1567, 0.1511, 0.1441, 0.1350, 0.1261,
        0.1180, 0.1103, 0.1035, 0.0982, 0.0933, 0.0878, 0.0811, 0.0752, 0.0709,
        0.0666, 0.0609, 0.0547, 0.0501, 0.0465, 0.0419, 0.0364, 0.0318, 0.0297,
        0.0299, 0.0323, 0.0365, 0.0412, 0.0453, 0.0493, 0.0542, 0.0602, 0.0658,
        0.0703, 0.0738, 0.0777, 0.0820, 0.0858, 0.0881, 0.0896, 0.0924, 0.0956,
        0.0973, 0.0962, 0.0937, 0.0916, 0.0909, 0.0902, 0.0883, 0.0864, 0.0862,
        0.0865, 0.0854, 0.0828, 0.0805, 0.0792, 0.0776, 0.0745, 0.0700, 0.0662,
        0.0637, 0.0612, 0.0573, 0.0529, 0.0495, 0.0470, 0.0446, 0.0423, 0.0406,
        0.0399, 0.0393, 0.0384, 0.0375, 0.0370, 0.0367, 0.0364, 0.0361, 0.0358,
        0.0358, 0.0360, 0.0361, 0.0360, 0.0361, 0.0364, 0.0364, 0.0360, 0.0351,
        0.0337, 0.0323, 0.0309, 0.0296, 0.0280, 0.0263, 0.0245, 0.0228, 0.0215,
        0.0207, 0.0201, 0.0198, 0.0195, 0.0194, 0.0193, 0.0192, 0.0191, 0.0191,
        0.0190, 0.0190, 0.0189, 0.0189, 0.0189, 0.0188, 0.0188, 0.0188, 0.0188,
        0.0188, 0.0188, 0.0187, 0.0187, 0.0187]
"""
noise = [0.3741, 0.5417, 0.7552, 0.7973, 0.7908, 0.7392, 0.7219, 0.7087, 0.6808,
        0.6472, 0.6215, 0.6107, 0.5729, 0.5517, 0.5393, 0.5028, 0.4972, 0.4932,
        0.4592, 0.4849, 0.4467, 0.4685, 0.4447, 0.4299, 0.4231, 0.4464, 0.3988,
        0.3910, 0.4093, 0.3671, 0.3742, 0.3559, 0.3713, 0.3513, 0.3473, 0.3253,
        0.3174, 0.3182, 0.3320, 0.2962, 0.3011, 0.2843, 0.3078, 0.2828, 0.2761,
        0.2861, 0.2634, 0.2668, 0.2577, 0.2488, 0.2728, 0.2536, 0.2378, 0.2427,
        0.2315, 0.2266, 0.2431, 0.2371, 0.2177, 0.2186, 0.2234, 0.2120, 0.2093,
        0.2199, 0.2115, 0.1999, 0.2024, 0.2117, 0.1999, 0.1930, 0.1925, 0.1946,
        0.1845, 0.1804, 0.1843, 0.1938, 0.1887, 0.1777, 0.1726, 0.1735, 0.1736,
        0.1652, 0.1624, 0.1637, 0.1700, 0.1677, 0.1580, 0.1541, 0.1544, 0.1556,
        0.1544, 0.1491, 0.1470, 0.1463, 0.1480, 0.1489, 0.1432, 0.1387, 0.1385,
        0.1403, 0.1428, 0.1414, 0.1355, 0.1333, 0.1321, 0.1318, 0.1324, 0.1305,
        0.1264, 0.1252, 0.1234, 0.1240, 0.1261, 0.1274, 0.1219, 0.1178, 0.1166,
        0.1158, 0.1153, 0.1153, 0.1142, 0.1114, 0.1100, 0.1087, 0.1077, 0.1072,
        0.1086, 0.1078, 0.1051, 0.1026, 0.1019, 0.1021, 0.1018, 0.1017, 0.1023,
        0.1001, 0.0981, 0.0976, 0.0970, 0.0954, 0.0952, 0.0956, 0.0953, 0.0936,
        0.0918, 0.0909, 0.0903, 0.0899, 0.0897, 0.0901, 0.0915, 0.0917, 0.0896,
        0.0879, 0.0868, 0.0861, 0.0853, 0.0848, 0.0846, 0.0842, 0.0838, 0.0827,
        0.0818, 0.0809, 0.0805, 0.0799, 0.0798, 0.0797, 0.0797, 0.0800, 0.0784,
        0.0773, 0.0765, 0.0759, 0.0752, 0.0747, 0.0741, 0.0733, 0.0730, 0.0734,
        0.0724, 0.0719, 0.0712, 0.0710, 0.0709, 0.0702, 0.0698, 0.0694, 0.0697,
        0.0696, 0.0694, 0.0686, 0.0682, 0.0671, 0.0669, 0.0667, 0.0667, 0.0665,
        0.0664, 0.0663, 0.0663, 0.0668, 0.0658, 0.0651, 0.0647, 0.0642, 0.0638,
        0.0636, 0.0633, 0.0632, 0.0632, 0.0629, 0.0627, 0.0626, 0.0617, 0.0614,
        0.0610, 0.0606, 0.0603, 0.0601, 0.0594, 0.0589, 0.0586, 0.0583, 0.0576,
        0.0566, 0.0553, 0.0537, 0.0515, 0.0492, 0.0467, 0.0438, 0.0409, 0.0377,
        0.0343, 0.0310, 0.0279, 0.0249, 0.0222, 0.0197, 0.0177, 0.0159, 0.0146,
        0.0135, 0.0128, 0.0123, 0.0119, 0.0117, 0.0115, 0.0115, 0.0114, 0.0113,
        0.0113, 0.0112, 0.0112, 0.0112, 0.0112]
noisy = [0.3741, 0.5417, 0.7552, 0.7973, 0.7908, 0.7392, 0.7219, 0.7087, 0.6808,
        0.6472, 0.6215, 0.6107, 0.5729, 0.5517, 0.5393, 0.5028, 0.4972, 0.4932,
        0.4592, 0.4849, 0.4467, 0.4685, 0.4447, 0.4299, 0.4231, 0.4464, 0.3988,
        0.3910, 0.4093, 0.3671, 0.3742, 0.3559, 0.3713, 0.3513, 0.3473, 0.3253,
        0.3174, 0.3182, 0.3320, 0.2962, 0.3011, 0.2843, 0.3078, 0.2828, 0.2761,
        0.2861, 0.2634, 0.2668, 0.2577, 0.2488, 0.2728, 0.2536, 0.2378, 0.2427,
        0.2315, 0.2266, 0.2431, 0.2371, 0.2177, 0.2186, 0.2234, 0.2120, 0.2093,
        0.2199, 0.2115, 0.1999, 0.2024, 0.2117, 0.1999, 0.1930, 0.1925, 0.1946,
        0.1845, 0.1804, 0.1843, 0.1938, 0.1887, 0.1777, 0.1726, 0.1735, 0.1736,
        0.1652, 0.1624, 0.1637, 0.1700, 0.1677, 0.1580, 0.1541, 0.1544, 0.1556,
        0.1544, 0.1491, 0.1470, 0.1463, 0.1480, 0.1489, 0.1432, 0.1387, 0.1385,
        0.1403, 0.1428, 0.1414, 0.1355, 0.1333, 0.1321, 0.1318, 0.1324, 0.1305,
        0.1264, 0.1252, 0.1234, 0.1240, 0.1261, 0.1274, 0.1219, 0.1178, 0.1166,
        0.1158, 0.1153, 0.1153, 0.1142, 0.1114, 0.1100, 0.1087, 0.1077, 0.1072,
        0.1086, 0.1078, 0.1051, 0.1026, 0.1019, 0.1021, 0.1018, 0.1017, 0.1023,
        0.1001, 0.0981, 0.0976, 0.0970, 0.0954, 0.0952, 0.0956, 0.0953, 0.0936,
        0.0918, 0.0909, 0.0903, 0.0899, 0.0897, 0.0901, 0.0915, 0.0917, 0.0896,
        0.0879, 0.0868, 0.0861, 0.0853, 0.0848, 0.0846, 0.0842, 0.0838, 0.0827,
        0.0818, 0.0809, 0.0805, 0.0799, 0.0798, 0.0797, 0.0797, 0.0800, 0.0784,
        0.0773, 0.0765, 0.0759, 0.0752, 0.0747, 0.0741, 0.0733, 0.0730, 0.0734,
        0.0724, 0.0719, 0.0712, 0.0710, 0.0709, 0.0702, 0.0698, 0.0694, 0.0697,
        0.0696, 0.0694, 0.0686, 0.0682, 0.0671, 0.0669, 0.0667, 0.0667, 0.0665,
        0.0664, 0.0663, 0.0663, 0.0668, 0.0658, 0.0651, 0.0647, 0.0642, 0.0638,
        0.0636, 0.0633, 0.0632, 0.0632, 0.0629, 0.0627, 0.0626, 0.0617, 0.0614,
        0.0610, 0.0606, 0.0603, 0.0601, 0.0594, 0.0589, 0.0586, 0.0583, 0.0576,
        0.0566, 0.0553, 0.0537, 0.0515, 0.0492, 0.0467, 0.0438, 0.0409, 0.0377,
        0.0343, 0.0310, 0.0279, 0.0249, 0.0222, 0.0197, 0.0177, 0.0159, 0.0146,
        0.0135, 0.0128, 0.0123, 0.0119, 0.0117, 0.0115, 0.0115, 0.0114, 0.0113,
        0.0113, 0.0112, 0.0112, 0.0112, 0.0112]
clean = [0.8367, 0.6600, 0.8598, 1.7670, 2.3002, 2.1797, 2.3497, 2.3408, 2.0383,
        1.8179, 1.7342, 1.7217, 1.6937, 1.6684, 1.6234, 1.4748, 1.3128, 1.1767,
        1.0691, 0.9546, 0.8632, 0.7860, 0.7011, 0.6398, 0.5863, 0.5318, 0.4918,
        0.4573, 0.4258, 0.4037, 0.3804, 0.3613, 0.3465, 0.3338, 0.3260, 0.3178,
        0.3101, 0.3013, 0.2927, 0.2877, 0.2807, 0.2748, 0.2728, 0.2688, 0.2670,
        0.2652, 0.2619, 0.2596, 0.2570, 0.2522, 0.2475, 0.2435, 0.2390, 0.2330,
        0.2273, 0.2214, 0.2152, 0.2090, 0.2032, 0.1966, 0.1909, 0.1857, 0.1803,
        0.1759, 0.1721, 0.1687, 0.1656, 0.1633, 0.1612, 0.1604, 0.1603, 0.1607,
        0.1608, 0.1618, 0.1623, 0.1629, 0.1640, 0.1645, 0.1650, 0.1651, 0.1640,
        0.1627, 0.1606, 0.1584, 0.1556, 0.1526, 0.1496, 0.1463, 0.1438, 0.1414,
        0.1388, 0.1362, 0.1338, 0.1316, 0.1300, 0.1284, 0.1269, 0.1257, 0.1246,
        0.1234, 0.1223, 0.1213, 0.1203, 0.1195, 0.1189, 0.1184, 0.1180, 0.1177,
        0.1172, 0.1167, 0.1160, 0.1153, 0.1144, 0.1136, 0.1128, 0.1121, 0.1109,
        0.1101, 0.1092, 0.1082, 0.1069, 0.1057, 0.1049, 0.1042, 0.1039, 0.1038,
        0.1036, 0.1029, 0.1010, 0.0990, 0.0975, 0.0964, 0.0955, 0.0949, 0.0940,
        0.0933, 0.0928, 0.0921, 0.0914, 0.0909, 0.0904, 0.0901, 0.0897, 0.0893,
        0.0890, 0.0886, 0.0883, 0.0879, 0.0873, 0.0871, 0.0868, 0.0866, 0.0864,
        0.0862, 0.0858, 0.0853, 0.0850, 0.0847, 0.0844, 0.0841, 0.0838, 0.0836,
        0.0834, 0.0831, 0.0830, 0.0829, 0.0827, 0.0824, 0.0823, 0.0821, 0.0817,
        0.0815, 0.0814, 0.0812, 0.0810, 0.0809, 0.0808, 0.0803, 0.0802, 0.0802,
        0.0802, 0.0800, 0.0797, 0.0794, 0.0792, 0.0789, 0.0787, 0.0785, 0.0782,
        0.0778, 0.0776, 0.0773, 0.0770, 0.0766, 0.0762, 0.0760, 0.0756, 0.0755,
        0.0753, 0.0753, 0.0751, 0.0750, 0.0750, 0.0749, 0.0748, 0.0746, 0.0745,
        0.0744, 0.0742, 0.0741, 0.0739, 0.0737, 0.0736, 0.0733, 0.0731, 0.0729,
        0.0727, 0.0725, 0.0720, 0.0715, 0.0710, 0.0704, 0.0697, 0.0688, 0.0678,
        0.0666, 0.0651, 0.0634, 0.0614, 0.0590, 0.0564, 0.0535, 0.0503, 0.0471,
        0.0438, 0.0405, 0.0374, 0.0344, 0.0317, 0.0293, 0.0273, 0.0256, 0.0242,
        0.0232, 0.0225, 0.0219, 0.0216, 0.0214, 0.0212, 0.0211, 0.0210, 0.0210,
        0.0209, 0.0209, 0.0209, 0.0209, 0.0208]
ssl_noise = [0.1522, 0.2127, 0.3432, 0.4478, 0.4920, 0.4518, 0.4511, 0.5263, 0.5725,
        0.5326, 0.4644, 0.4535, 0.4509, 0.4302, 0.3904, 0.3428, 0.3297, 0.3157,
        0.2821, 0.2818, 0.2436, 0.2422, 0.2258, 0.2213, 0.2364, 0.2817, 0.2724,
        0.2838, 0.3244, 0.3084, 0.3144, 0.2901, 0.3144, 0.3352, 0.3663, 0.3694,
        0.3869, 0.4354, 0.5083, 0.4895, 0.5170, 0.4918, 0.5508, 0.5396, 0.5663,
        0.5947, 0.5190, 0.5010, 0.4839, 0.4732, 0.5044, 0.4416, 0.3891, 0.3792,
        0.3428, 0.3133, 0.3084, 0.2728, 0.2272, 0.2097, 0.1962, 0.1687, 0.1502,
        0.1424, 0.1256, 0.1092, 0.1005, 0.0944, 0.0825, 0.0753, 0.0741, 0.0777,
        0.0814, 0.0891, 0.1022, 0.1221, 0.1353, 0.1456, 0.1601, 0.1809, 0.2016,
        0.2124, 0.2286, 0.2464, 0.2639, 0.2584, 0.2337, 0.2155, 0.2055, 0.1974,
        0.1873, 0.1747, 0.1693, 0.1656, 0.1650, 0.1638, 0.1553, 0.1457, 0.1393,
        0.1352, 0.1306, 0.1207, 0.1091, 0.1076, 0.1137, 0.1192, 0.1189, 0.1122,
        0.1077, 0.1136, 0.1202, 0.1219, 0.1179, 0.1143, 0.1151, 0.1214, 0.1238,
        0.1167, 0.1059, 0.0995, 0.0952, 0.0877, 0.0785, 0.0711, 0.0688, 0.0735,
        0.0840, 0.0935, 0.0998, 0.1062, 0.1166, 0.1296, 0.1397, 0.1459, 0.1500,
        0.1498, 0.1484, 0.1456, 0.1402, 0.1337, 0.1291, 0.1238, 0.1159, 0.1060,
        0.0972, 0.0899, 0.0838, 0.0790, 0.0751, 0.0706, 0.0656, 0.0604, 0.0554,
        0.0509, 0.0455, 0.0398, 0.0356, 0.0322, 0.0282, 0.0233, 0.0192, 0.0173,
        0.0174, 0.0194, 0.0229, 0.0268, 0.0303, 0.0337, 0.0381, 0.0434, 0.0476,
        0.0509, 0.0533, 0.0561, 0.0593, 0.0621, 0.0636, 0.0646, 0.0667, 0.0696,
        0.0700, 0.0688, 0.0664, 0.0649, 0.0644, 0.0635, 0.0619, 0.0602, 0.0604,
        0.0612, 0.0604, 0.0578, 0.0559, 0.0545, 0.0534, 0.0511, 0.0480, 0.0450,
        0.0431, 0.0412, 0.0383, 0.0352, 0.0321, 0.0300, 0.0281, 0.0262, 0.0249,
        0.0244, 0.0239, 0.0233, 0.0226, 0.0222, 0.0220, 0.0218, 0.0214, 0.0212,
        0.0211, 0.0213, 0.0213, 0.0213, 0.0213, 0.0216, 0.0217, 0.0215, 0.0209,
        0.0199, 0.0189, 0.0179, 0.0169, 0.0157, 0.0144, 0.0131, 0.0119, 0.0110,
        0.0104, 0.0100, 0.0098, 0.0096, 0.0095, 0.0094, 0.0094, 0.0093, 0.0093,
        0.0093, 0.0092, 0.0092, 0.0092, 0.0092, 0.0092, 0.0092, 0.0091, 0.0091,
        0.0091, 0.0091, 0.0091, 0.0091, 0.0091]
ssl_noisy = [0.5104, 0.5450, 0.7988, 1.4903, 1.9745, 1.8586, 2.0471, 2.4105, 2.4132,
        2.1505, 1.8908, 1.8712, 1.9302, 1.8830, 1.7155, 1.4804, 1.3089, 1.1602,
        1.0184, 0.9020, 0.7731, 0.6992, 0.6240, 0.5866, 0.5934, 0.6354, 0.6259,
        0.6302, 0.6672, 0.6549, 0.6387, 0.5908, 0.6122, 0.6542, 0.7111, 0.7309,
        0.7669, 0.8457, 0.9547, 0.9716, 1.0020, 0.9735, 1.0405, 1.0622, 1.1258,
        1.1578, 1.0519, 1.0007, 0.9816, 0.9686, 0.9717, 0.8759, 0.7924, 0.7521,
        0.6889, 0.6291, 0.5880, 0.5184, 0.4458, 0.4042, 0.3699, 0.3234, 0.2870,
        0.2641, 0.2360, 0.2101, 0.1922, 0.1778, 0.1598, 0.1487, 0.1463, 0.1521,
        0.1612, 0.1767, 0.1995, 0.2314, 0.2601, 0.2880, 0.3214, 0.3617, 0.4013,
        0.4323, 0.4662, 0.4955, 0.5138, 0.5004, 0.4622, 0.4272, 0.4022, 0.3810,
        0.3590, 0.3382, 0.3270, 0.3181, 0.3126, 0.3078, 0.2958, 0.2809, 0.2678,
        0.2569, 0.2452, 0.2270, 0.2089, 0.2075, 0.2192, 0.2296, 0.2284, 0.2170,
        0.2114, 0.2236, 0.2373, 0.2394, 0.2286, 0.2199, 0.2254, 0.2411, 0.2457,
        0.2317, 0.2102, 0.1968, 0.1882, 0.1746, 0.1574, 0.1432, 0.1392, 0.1486,
        0.1682, 0.1868, 0.1995, 0.2118, 0.2310, 0.2545, 0.2733, 0.2839, 0.2895,
        0.2912, 0.2908, 0.2848, 0.2739, 0.2627, 0.2535, 0.2423, 0.2269, 0.2098,
        0.1943, 0.1807, 0.1689, 0.1597, 0.1517, 0.1426, 0.1319, 0.1217, 0.1133,
        0.1053, 0.0951, 0.0843, 0.0762, 0.0698, 0.0619, 0.0525, 0.0446, 0.0409,
        0.0412, 0.0453, 0.0523, 0.0601, 0.0673, 0.0741, 0.0827, 0.0931, 0.1023,
        0.1095, 0.1151, 0.1213, 0.1284, 0.1345, 0.1381, 0.1404, 0.1450, 0.1508,
        0.1528, 0.1507, 0.1462, 0.1429, 0.1418, 0.1402, 0.1370, 0.1338, 0.1338,
        0.1349, 0.1331, 0.1283, 0.1244, 0.1218, 0.1194, 0.1144, 0.1073, 0.1009,
        0.0969, 0.0928, 0.0865, 0.0795, 0.0735, 0.0693, 0.0653, 0.0614, 0.0587,
        0.0575, 0.0565, 0.0551, 0.0537, 0.0528, 0.0524, 0.0520, 0.0512, 0.0508,
        0.0508, 0.0512, 0.0513, 0.0512, 0.0513, 0.0518, 0.0520, 0.0514, 0.0500,
        0.0478, 0.0455, 0.0434, 0.0412, 0.0387, 0.0359, 0.0330, 0.0304, 0.0284,
        0.0271, 0.0262, 0.0257, 0.0253, 0.0251, 0.0250, 0.0248, 0.0247, 0.0247,
        0.0246, 0.0245, 0.0245, 0.0244, 0.0244, 0.0244, 0.0243, 0.0243, 0.0243,
        0.0243, 0.0242, 0.0242, 0.0242, 0.0242]
ssl_clean = [0.4167, 0.4005, 0.5526, 1.1860, 1.6570, 1.5795, 1.7772, 2.0900, 2.0567,
        1.8193, 1.6065, 1.5931, 1.6543, 1.6193, 1.4757, 1.2709, 1.1018, 0.9571,
        0.8366, 0.7136, 0.6115, 0.5342, 0.4694, 0.4337, 0.4269, 0.4291, 0.4283,
        0.4223, 0.4230, 0.4258, 0.4022, 0.3740, 0.3727, 0.3986, 0.4300, 0.4488,
        0.4708, 0.5093, 0.5556, 0.5936, 0.5997, 0.5940, 0.6089, 0.6449, 0.6889,
        0.6952, 0.6546, 0.6164, 0.6123, 0.6091, 0.5807, 0.5378, 0.4981, 0.4633,
        0.4295, 0.3929, 0.3519, 0.3101, 0.2753, 0.2466, 0.2216, 0.1975, 0.1754,
        0.1575, 0.1429, 0.1305, 0.1190, 0.1089, 0.1005, 0.0952, 0.0937, 0.0965,
        0.1029, 0.1125, 0.1247, 0.1401, 0.1588, 0.1793, 0.2018, 0.2258, 0.2491,
        0.2727, 0.2941, 0.3090, 0.3120, 0.3027, 0.2849, 0.2642, 0.2464, 0.2308,
        0.2162, 0.2055, 0.1983, 0.1920, 0.1864, 0.1822, 0.1773, 0.1701, 0.1619,
        0.1539, 0.1453, 0.1350, 0.1265, 0.1261, 0.1331, 0.1389, 0.1378, 0.1317,
        0.1300, 0.1374, 0.1460, 0.1465, 0.1386, 0.1325, 0.1378, 0.1487, 0.1512,
        0.1428, 0.1298, 0.1214, 0.1161, 0.1085, 0.0986, 0.0903, 0.0882, 0.0938,
        0.1048, 0.1157, 0.1235, 0.1306, 0.1412, 0.1540, 0.1645, 0.1697, 0.1716,
        0.1736, 0.1743, 0.1703, 0.1636, 0.1577, 0.1521, 0.1450, 0.1359, 0.1269,
        0.1187, 0.1109, 0.1041, 0.0988, 0.0939, 0.0883, 0.0816, 0.0756, 0.0714,
        0.0670, 0.0612, 0.0550, 0.0504, 0.0468, 0.0422, 0.0366, 0.0320, 0.0298,
        0.0300, 0.0325, 0.0367, 0.0414, 0.0456, 0.0496, 0.0546, 0.0605, 0.0662,
        0.0707, 0.0743, 0.0781, 0.0825, 0.0863, 0.0886, 0.0901, 0.0930, 0.0962,
        0.0979, 0.0968, 0.0942, 0.0922, 0.0915, 0.0907, 0.0888, 0.0869, 0.0867,
        0.0871, 0.0860, 0.0833, 0.0810, 0.0797, 0.0781, 0.0750, 0.0705, 0.0666,
        0.0641, 0.0616, 0.0577, 0.0532, 0.0498, 0.0473, 0.0449, 0.0425, 0.0409,
        0.0401, 0.0396, 0.0387, 0.0378, 0.0372, 0.0370, 0.0366, 0.0363, 0.0360,
        0.0360, 0.0363, 0.0363, 0.0362, 0.0363, 0.0366, 0.0367, 0.0362, 0.0353,
        0.0339, 0.0325, 0.0311, 0.0298, 0.0282, 0.0264, 0.0246, 0.0229, 0.0217,
        0.0208, 0.0202, 0.0199, 0.0196, 0.0195, 0.0194, 0.0193, 0.0192, 0.0192,
        0.0191, 0.0191, 0.0190, 0.0190, 0.0190, 0.0189, 0.0189, 0.0189, 0.0189,
        0.0189, 0.0189, 0.0188, 0.0188, 0.0188]
"""
import matplotlib.pyplot as plt
Fs = 16000
N = 512
x = range(len(noise))
freqs = [i * (2.0*Fs/N)/1000.0 for i in x]

xticks = [i for i in range(0, 257, 16)]

fig, axs = plt.subplots(3, 3, figsize=(40,20))
fig.tight_layout()
axs[0,0].plot(x,noise)
axs[0,0].title.set_text("Noise")
axs[0,0].set_ylabel("Magnitude")
axs[0,0].set_xticks(xticks)
axs[0,0].set_xticklabels([str(freqs[x]) for x in xticks])
#axs[0,0].set_xlabel("Frequncy (kHz)")
axs[1,0].plot(x,clean, label='clean')
axs[1,0].title.set_text("Clean")
axs[1,0].set_ylabel("Magnitude")
axs[1,0].set_xticks(xticks)
axs[1,0].set_xticklabels([str(freqs[x]) for x in xticks])
#axs[1,0].set_xlabel("Frequncy (kHz)")
axs[2,0].plot(x,noisy)
axs[2,0].title.set_text("Noisy")
axs[2,0].set_ylabel("Magnitude")
axs[2,0].set_xticks(xticks)
axs[2,0].set_xticklabels([str(freqs[x]) for x in xticks])
axs[2,0].set_xlabel("Frequncy (kHz)")
axs[0,1].plot(x,ssl_noise)
axs[0,1].set_ylabel("Magnitude")
axs[0,1].title.set_text("Pinna Noise")
axs[0,1].set_xticks(xticks)
axs[0,1].set_xticklabels([str(freqs[x]) for x in xticks])
#axs[0,1].set_xlabel("Frequncy (kHz)")
axs[1,1].plot(x,ssl_clean)
axs[1,1].set_ylabel("Magnitude")
axs[1,1].title.set_text("Pinna Clean")
axs[1,1].set_xticks(xticks)
axs[1,1].set_xticklabels([str(freqs[x]) for x in xticks])
#axs[1,1].set_xlabel("Frequncy (kHz)")
axs[2,1].plot(x,ssl_noisy)
axs[2,1].set_ylabel("Magnitude")
axs[2,1].title.set_text("Pinna Noisy")
axs[2,1].set_xticks(xticks)
axs[2,1].set_xticklabels([str(freqs[x]) for x in xticks])
axs[2,1].set_xlabel("Frequncy (kHz)")
gain_noise = [ssl_noise[i] / noise[i] for i in range(len(noise))]
gain_noisy = [ssl_noisy[i] / noisy[i] for i in range(len(noisy))]
gain_clean = [ssl_clean[i] / clean[i] for i in range(len(clean))]
#maxN = max(gain_noise)
#maxY = max(gain_noisy)
#maxC = max(gain_clean)
#gain_noise = [gain_noise[i] / maxN for i in range(len(gain_noise))]
#gain_noisy = [x / maxY for x in gain_noisy]
#gain_clean = [x / maxC for x in gain_clean]
axs[0,2].plot(x,gain_noise, label='gain_noise')
axs[0,2].set_ylabel("Gain")
axs[0,2].title.set_text("Noise Gain")
axs[0,2].set_xticks(xticks)
axs[0,2].set_xticklabels([str(freqs[x]) for x in xticks])
axs[0,2].axhline(y=1.0,c="red",linewidth=0.5,zorder=0)
axs[1,2].plot(x,gain_clean, label='gain_clean')
axs[1,2].set_ylabel("Gain")
axs[1,2].title.set_text("Clean Gain")
axs[1,2].set_xticks(xticks)
axs[1,2].set_xticklabels([str(freqs[x]) for x in xticks])
axs[1,2].set_xticks(xticks)
axs[1,2].set_xticklabels([str(freqs[x]) for x in xticks])
axs[1,2].axhline(y=1.0,c="red",linewidth=0.5,zorder=0)
axs[2,2].plot(x,gain_noisy, label='gain_noisy')
axs[2,2].set_ylabel("Gain")
axs[2,2].title.set_text("Noisy Gain")
axs[2,2].set_xticks(xticks)
axs[2,2].set_xticklabels([str(freqs[x]) for x in xticks])
axs[2,2].axhline(y=1.0,c="red",linewidth=0.5,zorder=0)
axs[2,2].set_xlabel("Frequncy (kHz)")
plt.savefig("training_fft.png", bbox_inches='tight')
plt.close()
