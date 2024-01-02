dataset_paths = {
        #FFHQ and disease
	'genetic_dataset' : '/data/patelt6/FaceImages12Diseases09072022/TrimImg_align_1024pix_train_test/train/',
	'genetic_dataset_test' : '/data/patelt6/FaceImages12Diseases09072022/TrimImg_align_1024pix_train_test/test/',
        
        'ffhq_genetic' : '/data/patelt6/ffhq-dataset/FFHQ_256/images256x256/',
	'ffhq_genetic_test' : '/data/patelt6/ffhq-dataset/FFHQ_256/ffhq_and_genetic_disease_testset/',

        'ffhq_genetic_HQ' : '/data/patelt6/ffhq-dataset/FFHQ_1024/images1024x1024/',
	'ffhq_genetic_test_HQ' : '/data/patelt6/ffhq-dataset/FFHQ_1024/testset/',

	'cars_train': '',
	'cars_test': '',

	'celeba_train': '',
	'celeba_test': '',
	'celeba_test_w_inv': '',
	'celeba_test_w_latents': '',

	'ffhq': '',
	'ffhq_w_inv': '',
	'ffhq_w_latents': '',

	'afhq_wild_train': '',
	'afhq_wild_test': '',

}

model_paths = {
	# models for backbones and losses
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'resnet34': 'pretrained_models/resnet34-333f7ec4.pth',
	'moco': 'pretrained_models/moco_v2_800ep_pretrain.pt',
	# stylegan2 generators
	'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
	'stylegan_cars': 'pretrained_models/stylegan2-car-config-f.pt',
	'stylegan_ada_wild': 'pretrained_models/afhqwild.pt',
	# model for face alignment
	'shape_predictor': '/data/patelt6/hyperstyle/pretrained_models/shape_predictor_68_face_landmarks.dat',
	# models for ID similarity computation
	'curricular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
	# WEncoders for training on various domains
	'faces_w_encoder': 'pretrained_models/faces_w_encoder.pt',
	'cars_w_encoder': 'pretrained_models/cars_w_encoder.pt',
	'afhq_wild_w_encoder': 'pretrained_models/afhq_wild_w_encoder.pt',
	'e4e_w_encoder': 'pretrained_models/e4e_ffhq_encode.pt',
	# models for domain adaptation
	'restyle_e4e_ffhq': 'pretrained_models/restyle_e4e_ffhq_encode.pt',
	'stylegan_pixar': 'pretrained_models/pixar.pt',
	'stylegan_toonify': 'pretrained_models/ffhq_cartoon_blended.pt',
	'stylegan_sketch': 'pretrained_models/sketch.pt',
	'stylegan_disney': 'pretrained_models/disney_princess.pt'
}

edit_paths = {
	'age': 'editing/interfacegan_directions/age.pt',
	'smile': 'editing/interfacegan_directions/smile.pt',
        'ws_smile': 'editing/interfacegan_directions/ws_boundary.pt',
        '22q_smile': 'editing/interfacegan_directions/22q_boundary.pt',
        'ns_smile': 'editing/interfacegan_directions/ns_boundary.pt',
        'ns_new_smile': 'editing/interfacegan_directions/ns_new_boundary.pt',
        'ns_new_corrected_smile' : 'editing/interfacegan_directions/ns_new_corrected_smile.pt',
	'NS_new_95_percent_smile':  'editing/interfacegan_directions/NS_new_95_percent_smile.pt',
        'as_smile': 'editing/interfacegan_directions/AS_boundary.pt',
        'as_nosmile': 'editing/interfacegan_directions/AS_NoSmile_boundary.pt',
        'ws_nosmile': 'editing/interfacegan_directions/WS_NoSmile_boundary.pt',
	'ns_selected_95_percent_boundry': 'editing/interfacegan_directions/ns_selected_95_percent_boundry.pt',
        '22q_selected_1_boundary' : 'editing/interfacegan_directions/22q_selected_1_boundary.pt',
        '22q_new_corrected_boundary': 'editing/interfacegan_directions/22q_new_corrected_boundary.pt',
        '22q_least_boundary': 'editing/interfacegan_directions/22q_least_boundary.pt',
        'AS_selected_boundary': 'editing/interfacegan_directions/AS_selected_boundary.pt',
        'WS_No_Smile' : 'editing/interfacegan_directions/WS_No_Smile.pt',
        '22q_least_AddedUnaffected_boundary' : 'editing/interfacegan_directions/22q_least_AddedUnaffected_boundary.pt',
        'unaffected_to_KS' : 'editing/interfacegan_directions/unaffected_to_KS_boundary.pt',
        'KS_to_unaffected' : 'editing/interfacegan_directions/selected_KS_to_unaffected_boundary.pt',
	'pose': 'editing/interfacegan_directions/pose.pt',
	'cars': 'editing/ganspace_directions/cars_pca.pt',
	'styleclip': {
		'delta_i_c': 'editing/styleclip/global_directions/ffhq/fs3.npy',
		's_statistics': 'editing/styleclip/global_directions/ffhq/S_mean_std',
		'templates': 'editing/styleclip/global_directions/templates.txt'
	}
}