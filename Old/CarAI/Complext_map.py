        self.outer_track = 2*np.array( [ [ 10,  10 ],
                                         [ 10,  70 ],
                                         [ 15,  80 ],
                                         [ 20,  90 ],
                                         [ 50,  90 ],
                                         [ 80,  90 ],
                                         [ 90,  80 ],
                                         [ 90,  70 ],
                                         [ 80,  62 ],
                                         [ 50,  62 ],
                                         [ 44,  63 ],
                                         [ 44,  57 ],
                                         [ 50,  58 ],
                                         [ 80,  58 ],
                                         [ 90,  50 ],
                                         [ 75,  35 ],
                                         [ 90,  20 ],
                                         [ 90,  10 ] ] )


        self.inner_track = 2*np.array( [ [ 20,  20 ],
                                         [ 20,  70 ],
                                         [ 25,  80 ],
                                         [ 50,  80 ],
                                         [ 80,  80 ],
                                         [ 80,  70 ],
                                         [ 50,  70 ],
                                         [ 40,  75 ],
                                         [ 30,  65 ],
                                         [ 30,  55 ],
                                         [ 40,  45 ],
                                         [ 50,  50 ],
                                         [ 75,  50 ],
                                         [ 75,  45 ],
                                         [ 65,  35 ],
                                         [ 80,  20 ] ] )

        self.reward_gates = 2*np.array([ [ [10, 30], [20, 30] ],
                                         [ [10, 40], [20, 40] ],
                                         [ [10, 50], [20, 50] ],
                                         [ [10, 60], [20, 60] ],
                                         [ [10, 70], [20, 70] ],
                                         [ [20, 90], [25, 80] ],
                                         [ [35, 90], [35, 80] ],
                                         [ [45, 90], [45, 80] ],
                                         [ [55, 90], [55, 80] ],
                                         [ [65, 90], [65, 80] ],
                                         [ [80, 90], [80, 80] ],
                                         [ [90, 80], [80, 80] ],
                                         [ [80, 70], [90, 70] ],
                                         [ [80, 62], [80, 70] ],
                                         [ [70, 62], [70, 70] ],
                                         [ [60, 62], [60, 70] ],
                                         [ [50, 62], [50, 70] ],
                                         [ [44, 63], [40, 75] ],
                                         [ [44, 63], [30, 65] ],
                                         [ [30, 60], [44, 60] ],
                                         [ [44, 57], [30, 55] ],
                                         [ [44, 57], [40, 45] ],
                                         [ [50, 58], [50, 50] ],
                                         [ [60, 58], [60, 50] ],
                                         [ [70, 58], [70, 50] ],
                                         [ [90, 50], [75, 50] ],
                                         [ [75, 35], [65, 35] ],
                                         [ [75, 35], [65, 35] ],
                                         [ [80, 20], [80, 10] ],
                                         [ [80, 20], [80, 10] ],
                                         [ [65, 20], [65, 10] ],
                                         [ [55, 20], [55, 10] ],
                                         [ [45, 20], [45, 10] ],
                                         [ [35, 20], [35, 10] ],
                                         [ [20, 20], [20, 10] ], ] )
