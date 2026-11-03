"""Exact terminal ACK observation after the poller has joined; no resends."""


def observe_sync_return(context):
    ack = context.control.receive()
    if ack is None:
        return None
    context.terminal_control_ack = dict(sequence=ack[0], value=dict(ack[1]))
    if ack[1].get('phase') == 'failed':
        error = RuntimeError('Lua terminal heartbeat failure: '+str(ack[1].get('reason')))
        context._fail(error)
        raise error
    return ack[1].get('phase') == 'sync_return'
