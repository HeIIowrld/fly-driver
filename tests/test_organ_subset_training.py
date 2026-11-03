import torch
from torch import nn

from flybrain_drive.multiview_learning import configure_trainable_groups


def test_organ_subset_optimizer_preserves_all_other_groups_and_circuit():
    model = nn.Module()
    model.adapter = nn.Module()
    model.adapter.visual_adapter = nn.Linear(2, 2, bias=False)
    model.adapter.organ_adapters = nn.ModuleList([nn.Linear(2, 2, bias=False) for _ in range(3)])
    model.adapter.group_names = ('haltere_L', 'haltere_R', 'leg')
    model.readout = nn.Linear(2, 2, bias=False)
    model.circuit = nn.Linear(2, 2, bias=False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    parameters = configure_trainable_groups(model, ['organs'], ['haltere_L', 'haltere_R'])
    optimizer = torch.optim.SGD(parameters, lr=.1)
    data = torch.ones(1, 2)
    output = (model.adapter.visual_adapter(data)+model.readout(data)+model.circuit(data)
              +sum(layer(data) for layer in model.adapter.organ_adapters))
    output.sum().backward()
    optimizer.step()
    changed = {name for name, value in model.state_dict().items()
               if not torch.equal(value, before[name])}
    assert changed == {'adapter.organ_adapters.0.weight', 'adapter.organ_adapters.1.weight'}
