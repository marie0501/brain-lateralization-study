from voxel import Voxel
from itertools import product, groupby

class Subject:
    def __init__(self, subject_id, voxels):
        self.subject_id = subject_id
        self.voxels = voxels

    def __str__(self):
        return f"Subject(ID: {self.subject_id}, Voxels: {self.voxels})"

    def group_voxels_by_criteria(self, roi=[None],side=[None], eccentricity_range=[None], beta_frequency=[None]):
        grouped_voxels = []
        
        # Generate all possible combinations of criteria
        all_criteria_combinations = list(product(roi,side, eccentricity_range, beta_frequency))
        
        # Filter voxels based on criteria combinations
        for criteria_combination in all_criteria_combinations:
            roi_filter, side_filter, eccentricity_filter, beta_frequency_filter  = criteria_combination

            if beta_frequency_filter is not None and not isinstance(beta_frequency_filter, (list, tuple)):
                beta_frequency_filter = [beta_frequency_filter]

            filtered_voxels = [voxel for voxel in self.voxels if
                               (roi_filter is None or voxel.roi == roi_filter) and
                               (side is None or voxel.side == side_filter) and
                               (eccentricity_filter is None or eccentricity_filter[0] <= voxel.eccentricity < eccentricity_filter[1]) and
                               (beta_frequency_filter is None or any(freq in list(voxel.beta_values) for freq in beta_frequency_filter))]

            if filtered_voxels:
                grouped_voxels.append(filtered_voxels)

        return grouped_voxels


    def calculate_average_betas(subject, roi, eccentricity_range):
        # Filter voxels based on ROI and eccentricity range
        filtered_voxels = [voxel for voxel in subject.voxels if voxel.roi == roi and eccentricity_range[0] <= voxel.eccentricity < eccentricity_range[1]]
    
        # Group filtered voxels by frequencies
        sorted_voxels = sorted(filtered_voxels, key=lambda voxel: list(voxel.beta_values.keys()))
        grouped_voxels = {tuple(key): list(group) for key, group in groupby(sorted_voxels, key=lambda voxel: list(voxel.beta_values.keys()))}
    
        # Calculate average for each frequency
        average_betas = {}
        for frequencies, voxels in grouped_voxels.items():
            beta_sum = {freq: sum(voxel.beta_values[freq] for voxel in voxels) for freq in frequencies}
            num_voxels = len(voxels)
            average_betas[frequencies] = {freq: beta_sum[freq] / num_voxels for freq in frequencies}
    
        return average_betas

