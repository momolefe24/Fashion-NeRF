from VITON.Parser_Free.DM_VTON.opt.base_opt import BaseOptions

'''
# Stage 1: Train the Teacher warping module
!python train_pb_warp.py --project runs/train/DM-VTON_demo --name Teacher_warp \
--device 0 --align_corners --batch_size 18 --workers 16 --lr 0.00005 \
--niter 1 --niter_decay 1 --save_period 1 \
--print_step 200 --sample_step 1000 \
--dataroot ../dataset/VITON-Clean/VITON_traindata
'''

class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()

    def _add_args(self) -> None:
        super()._add_args()


        """  ============================================ EXPERIMENT SELECTION  ====================== """
        self.parser.add_argument('--run_number',type=int, default=1, help='Which run number are we currently training')
        self.parser.add_argument('--experiment_number',type=int, default=1, help='Which experiment number are we currently training')
        self.parser.add_argument('--experiment_run',type=str, default="experiment_{}/run_{}", help='experiment_{experiment_number}/run_{run_number}')

        self.parser.add_argument('--run_from_number',type=int, default=1, help='Which run number are we loading checkpoints from')
        self.parser.add_argument('--experiment_from_number',type=int, default=1, help='Which experiment number are we loading checkpoints from')
        self.parser.add_argument('--experiment_from_run',type=str, default="experiment_{}/run_{}", help='experiment_{experiment_from_number}/{run_from_number}')

        """  ----------------- CONFIGURABLE FOR SELECTION DATA = CAN CHOOSE Vanilla_NeRF ----------------- """ 

        self.parser.add_argument('--VITON_Type', 
                    type=str, help='Which type are we training',
                    default='Parser_Free'
                    )
        # low_res_viton_dataset_name
        self.parser.add_argument('--low_res_viton_dataset_name', 
                    type=str, help='Which type are we training',
                    default='VITON-Clean'
                    )
        self.parser.add_argument('--VITON_Name', 
                    type=str, help='Which model are we training',
                    default='DM_VTON'
                    )
        self.parser.add_argument('--VITON_selection_dir', 
                    type=str, help='Which model are we training',
                    default='VITON/{}/{}'
                    )
        """  ============================================ NAME  ====================== """
        self.parser.add_argument('--res',type=str, help='Resolution',default='low_res')
        self.parser.add_argument('--dataset_name', default='Original')
        self.parser.add_argument("--training_method", default="warping")


        self.parser.add_argument("--gpu_ids", default = "")

        self.parser.add_argument('-b', '--batch-size', type=int, default=4)

        """  ----------------- CONFIGURABLE FOR SELECTION DATA = CAN CHOOSE RAIL | ORIGINAL ----------------- """
        self.parser.add_argument("--try_on_fine_height", type=int, default=1024)
        self.parser.add_argument("--datamode", default="train")
        self.parser.add_argument('--rail_dir', 
                    type=str, help='Data directory',
                    default='data/VITON/{}/processed/{}/{}'
                    )

        self.parser.add_argument('--original_dir', 
                        type=str, help='Data directory',
                        default='data/VITON/{}/processed/{}/{}'
                        )   
        self.parser.add_argument('--root_dir',type=str, help='Root directory', default='/home-mscluster/mmolefe/Playground/Synthesising Virtual Fashion Try-On with Neural Radiance Fields/Fashion-SuperNeRF')

        self.parser.add_argument("--model_dir", default="VITON/Parser_Free/DM_VTON/{}")
        self.parser.add_argument("--dataroot", default="../data/VITON-Clean")        
        self.parser.add_argument('--valroot', type=str, default='../data/VITON-Clean', help='val/test dataset path')

        self.parser.add_argument('--tensorboard_dir', type=str,
                        default='./tensorboard/{}/{}', help='inference_pipeline/tensorboard/experiment_1/run_1/VITON/parser-method/model')
        self.parser.add_argument('--results_dir', type=str,
                        default='./results/{}/{}', help='inference_pipeline/results/experiment_1/run_1/VITON/parser-method/model')
        
        
        """  ============================================ PB Gen ====================== """
        self.parser.add_argument('--pb_gen_save_step_checkpoint_dir', type=str,
                        default='./checkpoints/{}/{}/steps',help='inference_pipeline/checkpoints/experiment_1/run_1/VITON/parser-method/model/steps')
        self.parser.add_argument('--pb_gen_save_step_checkpoint', type=str, default='pb_gen_%06d.pt',
                        help='model checkpoint for initialization')

        self.parser.add_argument('--pb_gen_load_step_checkpoint_dir', type=str,
                        default='./checkpoints/{}/{}/steps',help='inference_pipeline/checkpoints/experiment_1/run_1/VITON/parser-method/model/steps')
        self.parser.add_argument('--pb_gen_load_step_checkpoint', type=str, default='pb_gen_%06d.pt',
                        help='model checkpoint for initialization')



        self.parser.add_argument('--pb_gen_save_final_checkpoint_dir', type=str,
                        default='./checkpoints/{}/{}',help='inference_pipeline/checkpoints/experiment_1/run_1/VITON/parser-method/model')
        self.parser.add_argument('--pb_gen_save_final_checkpoint', type=str, default='pb_gen.pt',
                        help='model checkpoint for initialization')

        self.parser.add_argument('--pb_gen_load_final_checkpoint_dir', type=str,
                        default='./checkpoints/{}/{}',help='inference_pipeline/checkpoints/experiment_1/run_1/VITON/parser-method/model')
        self.parser.add_argument('--pb_gen_load_final_checkpoint', type=str, default='pb_gen.pt',
                        help='model checkpoint for initialization')
        
        
        
        
        """  ============================================ PF Gen ====================== """
        self.parser.add_argument('--pf_gen_save_step_checkpoint_dir', type=str,
                        default='./checkpoints/{}/{}/steps',help='inference_pipeline/checkpoints/experiment_1/run_1/VITON/parser-method/model/steps')
        self.parser.add_argument('--pf_gen_save_step_checkpoint', type=str, default='pf_gen_%06d.pt',
                        help='model checkpoint for initialization')

        self.parser.add_argument('--pf_gen_load_step_checkpoint_dir', type=str,
                        default='./checkpoints/{}/{}/steps',help='inference_pipeline/checkpoints/experiment_1/run_1/VITON/parser-method/model/steps')
        self.parser.add_argument('--pf_gen_load_step_checkpoint', type=str, default='pf_gen_%06d.pt',
                        help='model checkpoint for initialization')



        self.parser.add_argument('--pf_gen_save_final_checkpoint_dir', type=str,
                        default='./checkpoints/{}/{}',help='inference_pipeline/checkpoints/experiment_1/run_1/VITON/parser-method/model')
        self.parser.add_argument('--pf_gen_save_final_checkpoint', type=str, default='pf_gen.pt',
                        help='model checkpoint for initialization')

        self.parser.add_argument('--pf_gen_load_final_checkpoint_dir', type=str,
                        default='./checkpoints/{}/{}',help='inference_pipeline/checkpoints/experiment_1/run_1/VITON/parser-method/model')
        self.parser.add_argument('--pf_gen_load_final_checkpoint', type=str, default='pf_gen.pt',
                        help='model checkpoint for initialization')
        
        
        
        
        
        
        
        
        
        
        
        """  ============================================ PB Warp ====================== """
        self.parser.add_argument('--pb_warp_save_step_checkpoint_dir', type=str,
                        default='./checkpoints/{}/{}/steps',help='inference_pipeline/checkpoints/experiment_1/run_1/VITON/parser-method/model/steps')
        self.parser.add_argument('--pb_warp_save_step_checkpoint', type=str, default='pb_warp_%06d.pt',
                        help='model checkpoint for initialization')

        self.parser.add_argument('--pb_warp_load_step_checkpoint_dir', type=str,
                        default='./checkpoints/{}/{}/steps',help='inference_pipeline/checkpoints/experiment_1/run_1/VITON/parser-method/model/steps')
        self.parser.add_argument('--pb_warp_load_step_checkpoint', type=str, default='pb_warp_%06d.pt',
                        help='model checkpoint for initialization')



        self.parser.add_argument('--pb_warp_save_final_checkpoint_dir', type=str,
                        default='./checkpoints/{}/{}',help='inference_pipeline/checkpoints/experiment_1/run_1/VITON/parser-method/model')
        self.parser.add_argument('--pb_warp_save_final_checkpoint', type=str, default='pb_warp.pt',
                        help='model checkpoint for initialization')

        self.parser.add_argument('--pb_warp_load_final_checkpoint_dir', type=str,
                        default='./checkpoints/{}/{}',help='inference_pipeline/checkpoints/experiment_1/run_1/VITON/parser-method/model')
        self.parser.add_argument('--pb_warp_load_final_checkpoint', type=str, default='pb_warp.pt',
                        help='model checkpoint for initialization')
        
        """  ============================================ PF Warp ====================== """
        self.parser.add_argument('--pf_warp_save_step_checkpoint_dir', type=str,
                        default='./checkpoints/{}/{}/steps',help='inference_pipeline/checkpoints/experiment_1/run_1/VITON/parser-method/model/steps')
        self.parser.add_argument('--pf_warp_save_step_checkpoint', type=str, default='pf_warp_%06d.pt',
                        help='model checkpoint for initialization')

        self.parser.add_argument('--pf_warp_load_step_checkpoint_dir', type=str,
                        default='./checkpoints/{}/{}/steps',help='inference_pipeline/checkpoints/experiment_1/run_1/VITON/parser-method/model/steps')
        self.parser.add_argument('--pf_warp_load_step_checkpoint', type=str, default='pf_warp_%06d.pt',
                        help='model checkpoint for initialization')



        self.parser.add_argument('--pf_warp_save_final_checkpoint_dir', type=str,
                        default='./checkpoints/{}/{}',help='inference_pipeline/checkpoints/experiment_1/run_1/VITON/parser-method/model')
        self.parser.add_argument('--pf_warp_save_final_checkpoint', type=str, default='pf_warp.pt',
                        help='model checkpoint for initialization')

        self.parser.add_argument('--pf_warp_load_final_checkpoint_dir', type=str,
                        default='./checkpoints/{}/{}',help='inference_pipeline/checkpoints/experiment_1/run_1/VITON/parser-method/model')
        self.parser.add_argument('--pf_warp_load_final_checkpoint', type=str, default='pf_warp.pt',
                        help='model checkpoint for initialization')
        # For logging
        self.parser.add_argument(
        '--print_step',
        type=int,
        default=100,
        help='frequency of print training results on screen',
        )
        self.parser.add_argument(
        '--sample_step', type=int, default=100, help='frequency of sample training results'
        )
        self.parser.add_argument(
        '--save_period',
        type=int,
        default=20,
        help='frequency of saving checkpoints at the end of epochs',
        )

        # For training
        self.parser.add_argument(
        '--resume', nargs='?', const=True, default=False, help='resume training'
        )
        self.parser.add_argument(
        '--use_dropout', action='store_true', help='use dropout for the generator'
        )
        self.parser.add_argument(
        '--align_corners', action='store_true', help='align corners for grid_sample'
        )
        self.parser.add_argument(
        '--verbose', action='store_true', default=False, help='toggles verbose'
        )
        self.parser.add_argument('--local_rank', type=int, default=-1)
        self.parser.add_argument(
        '--optimizer',
        type=str,
        choices=['SGD', 'Adam', 'AdamW'],
        default='Adam',
        help='optimizer',
        )
        self.parser.add_argument(
        '--niter', type=int, default=50, help='number of epochs at starting learning rate'
        )
        self.parser.add_argument(
        '--niter_decay',
        type=int,
        default=50,
        help='number of epochs to linearly decay learning rate to zero',
        )
        self.parser.add_argument(
        '--momentum', type=float, default=0.5, help='momentum term of optimizer'
        )
        self.parser.add_argument('--lr', type=float, default=0.00005, help='initial learning rate')

        # Checkpoints
        self.parser.add_argument(
        '--pb_warp_checkpoint',
        type=str,
        help='load the pretrained model from the specified location',
        )
        self.parser.add_argument(
        '--pb_gen_checkpoint',
        type=str,
        help='load the pretrained model from the specified location',
        )
        self.parser.add_argument(
        '--pf_warp_checkpoint',
        type=str,
        help='load the pretrained model from the specified location',
        )
        self.parser.add_argument(
        '--pf_gen_checkpoint',
        type=str,
        help='load the pretrained model from the specified location',
        )
