from universal_computation.fpt_antibias import FPTAntiBias

experiment_params = dict(
    model_name='gpt2',
    input_max_dim=50,
    pretrained=True,
    freeze_trans=True,  # if False, we don't check arguments other than in and out
    freeze_linear=False,
    freeze_pos=False,
    freeze_ln=False,
    freeze_attn=True,
    freeze_ff=True,
    freeze_out=False,
    linear_layer_sizes=None,  # not in paper, but can specify layer sizes for an MLP,
    # ex. [32, 32] creates a 2-layer MLP with dimension 32
    out_layer_sizes=None,
    learning_rate=1e-3,
    batch_size=2,
    dropout=0.1,
    orth_gain=1.41,
    position_ids=None,
    return_last_only=True,
)


def antibias_model(device):
    return FPTAntiBias(
        input_max_dim=experiment_params['input_max_dim'],
        model_name=experiment_params['model_name'],
        pretrained=experiment_params['pretrained'],
        return_last_only=experiment_params['return_last_only'],
        linear_layer_sizes=experiment_params['linear_layer_sizes'],
        out_layer_sizes=experiment_params['out_layer_sizes'],
        freeze_trans=experiment_params['freeze_trans'],
        freeze_linear=experiment_params['freeze_linear'],
        freeze_pos=experiment_params['freeze_pos'],
        freeze_ln=experiment_params['freeze_ln'],
        freeze_attn=experiment_params['freeze_attn'],
        freeze_ff=experiment_params['freeze_ff'],
        freeze_out=experiment_params['freeze_out'],
        position_ids=experiment_params['position_ids'],
        dropout=experiment_params['dropout'],
        orth_gain=experiment_params['orth_gain'],
        device=device
    ).to(device)
