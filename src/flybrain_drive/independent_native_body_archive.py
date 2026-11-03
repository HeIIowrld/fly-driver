"""Continuously drain an owned read-only native process into a finite async log.

Preserve original copy brackets and receipts. Repeated physics packets remain
explicit; downstream body history must retain each packet's first receipt.
"""
import threading

from .async_event_archive import AsyncEventArchive
from .native_process_sampler import NativeProcessSampler


class IndependentNativeBodyArchive:
    def __init__(self,path,*,recorded=None,maximum_samples=120000):
        self.archive=AsyncEventArchive(path)
        self.sampler=NativeProcessSampler(period=.005,max_samples=maximum_samples,
            capacity=128,max_age=.05,recorded=recorded)
        self.closing=threading.Event()
        self.error=None
        self.samples=0
        self.last_packet=None
        self.last_receipt=None
        self.maximum_receipt_gap=0.
        self.distinct_packets=set()
        try:
            self.sampler.start()
        except BaseException:
            try:
                self.sampler.stop()
            finally:
                self.archive.close()
            raise
        self.thread=threading.Thread(target=self._drain,name='native-body-archive',daemon=True)
        self.thread.start()

    def _drain(self):
        try:
            while True:
                try:
                    row=self.sampler.poll(timeout=.01)
                except RuntimeError as error:
                    terminal=self.sampler.terminal
                    if (self.closing.is_set() and terminal and terminal[0]=='stopped'
                            and self.sampler.failure is None
                            and str(error).startswith('producer terminated:')):
                        if self.sampler.expected==terminal[1]:
                            break
                        continue  # Drain every original row preceding the terminal count.
                    raise
                if row is None:
                    continue
                authority=row['authority'];packet=authority['packet_id']
                receipt=authority['received_monotonic_seconds']
                if (authority['builtin_ai_controlled'] or self.last_packet is not None and
                        (packet<self.last_packet or receipt<=self.last_receipt)):
                    raise ValueError('Native body authority/packet/receipt continuity failed')
                if self.last_receipt is not None:
                    self.maximum_receipt_gap=max(self.maximum_receipt_gap,receipt-self.last_receipt)
                self.archive.emit(dict(sequence=self.samples,**row))
                self.samples+=1
                self.distinct_packets.add(packet)
                self.last_packet,self.last_receipt=packet,receipt
        except BaseException as error:
            self.error=error

    def check(self):
        self.archive.check()
        if self.error:
            raise RuntimeError('Independent native body archive failed') from self.error

    def close(self):
        self.closing.set();self.sampler.stop_event.set()
        self.thread.join(5)
        forced=False
        try:
            if self.thread.is_alive():
                raise RuntimeError('Native body drain did not terminate')
            self.sampler.stop()
        except Exception as error:
            self.error=self.error or error
            if self.sampler.process.is_alive():
                self.sampler.process.terminate();self.sampler.process.join(5)
                forced=True
        try:
            self.archive.close()
        except Exception as error:
            self.error=self.error or error
        terminal=self.sampler.terminal
        complete=bool(not self.error and not forced and terminal and terminal[0]=='stopped'
            and terminal[1]==self.samples==self.archive.accepted==self.archive.written)
        return dict(status='completed' if complete else 'failed',samples=self.samples,
            distinct_packets=len(self.distinct_packets),terminal=terminal,
            maximum_copy_receipt_gap_ms=self.maximum_receipt_gap*1000,
            process_exitcode=self.sampler.process.exitcode,
            process_alive=self.sampler.process.is_alive(),drain_alive=self.thread.is_alive(),
            forced_termination=forced,error=str(self.error) if self.error else None,
            archive=self.archive.statistics())
