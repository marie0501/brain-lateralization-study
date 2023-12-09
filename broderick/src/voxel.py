class Voxel:
    def __init__(self, region_of_interest, eccentricity, side, beta_values):
        self.roi = region_of_interest
        self.eccentricity = eccentricity
        self.side = side
        self.beta_values = beta_values

    def __str__(self):
        return f"Voxel(ROI: {self.roi}, Eccentricity: {self.eccentricity}, Side: {self.side}, Beta Values: {self.beta_values})"