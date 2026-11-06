"""Single-producer source-bound decision mailbox; no Torch, file log or inference."""
import hashlib,json,struct,uuid
from multiprocessing.shared_memory import SharedMemory
HEADER=struct.Struct('<QII32s')
CAPACITY=16384

class FlightDecisionMailbox:
    def __init__(self,spec=None):
        self.owned=spec is None
        self.memory=SharedMemory(create=True,size=HEADER.size+CAPACITY+1,name='flybrain_decision_'+uuid.uuid4().hex) if self.owned else SharedMemory(name=spec['name'],create=False)
        self.spec=dict(name=self.memory.name,size=HEADER.size+CAPACITY+1,schema='flybrain.source_decision_mailbox.v1')
        if len(self.memory.buf)<self.spec['size'] or spec is not None and spec!=self.spec:
            self.memory.close();raise ValueError('Exact owned mailbox specification required')
        self.sequence=0
        if self.owned:self.memory.buf[:]=bytes(len(self.memory.buf))
    def publish(self,decision):
        payload=json.dumps(decision.record(),separators=(',',':'),allow_nan=False).encode()
        if not 0<len(payload)<=CAPACITY:raise ValueError('Bounded original envelope required')
        self.sequence+=1;odd=2*self.sequence-1;even=odd+1;digest=hashlib.sha256(payload).digest()
        self.memory.buf[:HEADER.size]=HEADER.pack(odd,1,len(payload),digest)
        self.memory.buf[HEADER.size:HEADER.size+len(payload)]=payload
        self.memory.buf[:HEADER.size]=HEADER.pack(even,1,len(payload),digest)
    def seal(self):
        self.sequence+=1;self.memory.buf[:HEADER.size]=HEADER.pack(2*self.sequence,2,0,bytes(32))
    def read(self,after_decision=0):
        from .current_flight_control_contract import FlightDecisionEnvelope
        for _ in range(8):
            header=bytes(self.memory.buf[:HEADER.size]);sequence,kind,size,digest=HEADER.unpack(header)
            if not sequence or sequence%2:return None
            if kind==2:raise RuntimeError('Decision producer terminal; buffered control forbidden')
            if kind!=1 or not 0<size<=CAPACITY:raise ValueError('Mailbox header corrupt')
            first=bytes(self.memory.buf[HEADER.size:HEADER.size+size]);second=bytes(self.memory.buf[HEADER.size:HEADER.size+size]);end=bytes(self.memory.buf[:HEADER.size])
            if header!=end or first!=second:continue
            if hashlib.sha256(first).digest()!=digest:raise ValueError('Mailbox payload digest differs')
            value=json.loads(first)
            for k in ['view_render_lowers','view_render_uppers','view_body_first_copy_starts','prediction']:value[k]=tuple(value[k])
            decision=FlightDecisionEnvelope(**value)
            if sequence!=2*decision.decision_id:raise ValueError('Mailbox sequence differs from original decision')
            return None if decision.decision_id<=after_decision else decision
        return None
    def request_stop(self):self.memory.buf[HEADER.size+CAPACITY]=1
    @property
    def stop_requested(self):return bool(self.memory.buf[HEADER.size+CAPACITY])
    def close(self):self.memory.close()
    def unlink(self):
        if not self.owned:raise ValueError('Only creator may unlink mailbox after children exit')
        self.memory.unlink()

class MailboxTrial:
    """GPU caller only arms/stops; the independent actuator consumes decisions."""
    def __init__(self,actuator,planned_seconds):self.actuator=actuator;self.planned_seconds=planned_seconds;self.error=None;self.decisions=[];self.attempts=[]
    def start(self,*,prepared_since,valid_camera_groups):return self.actuator.arm(prepared_since=prepared_since,valid_camera_groups=valid_camera_groups,planned_seconds=self.planned_seconds)
    def stop(self):pass


def consume_mailbox_step(lease, mailbox, guard, planned_end):
    """Check old lease, owned stop and plant guards before reading/applying fresh control."""
    lease.poll()
    if lease.state!='armed':return
    if lease.clock()>=planned_end:lease.stop('planned_observation_complete');return
    if mailbox.stop_requested:lease.stop('explicit_stop');return
    guard(lease.clock())
    lease.poll()
    if lease.state!='armed':return
    if lease.clock()>=planned_end:lease.stop('planned_observation_complete');return
    if mailbox.stop_requested:lease.stop('explicit_stop');return
    decision=mailbox.read(lease.last_decision)
    if decision is not None:lease.apply(decision)
