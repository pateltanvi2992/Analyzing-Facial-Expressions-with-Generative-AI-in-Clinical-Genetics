import torch

from configs.paths_config import edit_paths
from utils.common import tensor2im


class FaceEditor:

    def __init__(self, stylegan_generator):
        self.generator = stylegan_generator
        self.interfacegan_directions = {
            'age': torch.load(edit_paths['age']).cuda(),
            'smile': torch.load(edit_paths['smile']).cuda(),
	    'ws_smile': torch.load(edit_paths['ws_smile']).cuda(),
            '22q_smile': torch.load(edit_paths['22q_smile']).cuda(),
	    'ns_smile': torch.load(edit_paths['ns_smile']).cuda(),
            'as_smile': torch.load(edit_paths['as_smile']).cuda(),
            'ns_new_smile': torch.load(edit_paths['ns_new_smile']).cuda(),
	    'NS_new_95_percent_smile': torch.load(edit_paths['NS_new_95_percent_smile']).cuda(),
            'ns_selected_95_percent_boundry': torch.load(edit_paths['ns_selected_95_percent_boundry']).cuda(),
	    '22q_selected_1_boundary': torch.load(edit_paths['22q_selected_1_boundary']).cuda(),
            '22q_new_corrected_boundary': torch.load(edit_paths['22q_new_corrected_boundary']).cuda(),
            '22q_least_boundary' : torch.load(edit_paths['22q_least_boundary']).cuda(),
            '22q_least_AddedUnaffected_boundary' : torch.load(edit_paths['22q_least_AddedUnaffected_boundary']).cuda(),
	    'AS_selected_boundary' : torch.load(edit_paths['AS_selected_boundary']).cuda(),
            'unaffected_to_KS' : torch.load(edit_paths['unaffected_to_KS']).cuda(),
            'KS_to_unaffected' : torch.load(edit_paths['KS_to_unaffected']).cuda(),
            'WS_No_Smile' : torch.load(edit_paths['WS_No_Smile']).cuda(),
            'pose': torch.load(edit_paths['pose']).cuda()
        }

    def apply_interfacegan(self, latents, weights_deltas, direction, factor=1, factor_range=None):
        edit_latents = []
        direction = self.interfacegan_directions[direction]
        if factor_range is not None:  # Apply a range of editing factors. for example, (-5, 5)
            for f in range(*factor_range):
                edit_latent = latents + (10*f) * direction
                edit_latents.append(edit_latent)
            edit_latents = torch.stack(edit_latents).transpose(0,1)
        else:
            edit_latents = latents + factor * direction
        return self._latents_to_image(edit_latents, weights_deltas)

    def _latents_to_image(self, all_latents, weights_deltas):
        sample_results = {}
        with torch.no_grad():
            for idx, sample_latents in enumerate(all_latents):
                sample_deltas = [d[idx] if d is not None else None for d in weights_deltas]
                images, _ = self.generator([sample_latents],
                                           weights_deltas=sample_deltas,
                                           randomize_noise=False,
                                           input_is_latent=True)
                sample_results[idx] = [tensor2im(image) for image in images]
        return sample_results
