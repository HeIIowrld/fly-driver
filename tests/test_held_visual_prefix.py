import pytest
import torch
from flybrain_drive.held_visual_prefix import HeldVisualPrefix


class Network:
    def simulate(self, movie, *, dt, initial_state, as_states):
        result=[];state=initial_state
        for _ in range(movie.shape[1]):
            state=state+dt*movie[:,0,0,:]
            result.append(state)
        return result


@pytest.mark.parametrize('steps',[8,16])
@pytest.mark.parametrize('elapsed',[.002,.005,.0237,.08,.0923,.15])
def test_prefix_commits_only_elapsed_ticks_and_fraction(elapsed,steps):
    net=Network();initial=torch.zeros(1,1);held=torch.ones(1,1)
    prefix=HeldVisualPrefix(net,initial,held,steps=steps)
    assert prefix.commit(elapsed).item()==pytest.approx(elapsed,abs=1e-7)
    assert initial.item()==0
    with pytest.raises(RuntimeError,match='already consumed'):prefix.commit(elapsed)


def test_prefix_rejects_invalid_time_without_consuming():
    prefix=HeldVisualPrefix(Network(),torch.zeros(1,1),torch.ones(1,1))
    for bad in (0.,.151,float('nan'),True):
        with pytest.raises(ValueError):prefix.commit(bad)
    assert prefix.commit(.01).item()==pytest.approx(.01)
