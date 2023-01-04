
PENDING = object()
class Event:
    """An event that may happen at some point in time.
    An event
    - may happen (:attr:`triggered` is ``False``),
    - is going to happen (:attr:`triggered` is ``True``) or
    - has happened (:attr:`processed` is ``True``)."""
    def __init__(self, env):

        self.env = env          # The class:Environment the event lives in.
        self.callbacks = []     # List of functions that are called when the event is processed.
        self._value = PENDING
    @property
    def triggered(self):
        """Becomes True if the event has been triggered and its callbacks are about to be invoked."""
        return self._value is not PENDING
    @property
    def processed(self):
        """Becomes True if the event has been processed (e.g., its callbacks have been invoked)."""
        return self.callbacks is None
    @property
    def value(self):
        """The value of the event if it is available.
        The value is available when the event has been triggered.
        Raises :exc:`AttributeError` if the value is not yet available.
        """
        if self._value is PENDING:
            raise AttributeError('Value of %s is not yet available' % self)
        return self._value
    def trigger(self, value=None):
        """Trigger the event with the value provided.
        Raises :exc:RuntimeError if this event has already been triggerd.
        """
        if self._value is not PENDING:
            raise RuntimeError('%s has already been triggered' % self)

        self._ok = True
        self._value = value
        self.env.schedule(self)
        return self

class Enviroment:
    def __init__(self, initial_time=0):
        self._now = initial_time
        self._queue = []  # The list of all currently scheduled events.
        self._eid = count()  # Counter for event IDs
        self._active_proc = None
    @property
    def now(self):
        """The current simulation time."""
        return self._now
    def schedule(self, event, delay=0):
        """Schedule an *event* with a given *delay*."""
        heappush(self._queue, (self._now + delay, next(self._eid), event))
    def peek(self):
        """Get the time of the next scheduled event. 
        Return infinity if there is no further event."""
        try:
            return self._queue[0][0]
        except IndexError:
            return float('inf')
    def run(self, until=None):
        """Executes :meth:`step()` until the given criterion *until* is met.
        - If it is ``None`` (which is the default), this method will return
          when there are no further events to be processed.
        - If it is a number, the method will continue stepping
          until the environment's time reaches *until*.
        """
        at = float(until)

        if at <= self.now:
            raise ValueError('until(=%s) should be > the current simulation time.' % at)

        # Schedule the event before all regular timeouts.
        until = Event(self)
        until._ok = True
        until._value = None
        self.schedule(until, at - self.now)            

        try:
            while True:
                self.step()
        except: 
            if until is not None:
                assert not until.triggered
                raise RuntimeError('No scheduled events left but "until" event was not triggered: %s' % until)
    def step(self):
        """Process the next event. Raise an RuntimeError if no further events are available. """
        try:
            self._now, _, event = heappop(self._queue)
        except IndexError:
            raise RuntimeError('No scheduled events left')

        # Process callbacks of the event. Set the events callbacks to None
        # immediately to prevent concurrent modifications.
        callbacks, event.callbacks = event.callbacks, None
        for callback in callbacks:
            callback(event)
    # def timeout(self,event:Event,delay,value=None):
    #     if delay < 0:
    #         raise ValueError('Negative delay %s' % delay)
    #     event.callbacks = []
    #     event._value = value
    #     event._delay = delay
    #     event._ok = True
    #     self.schedule(event, delay)
    # def initialize(self,event:Event,process):
    #     event.callbacks = [process._resume]
    #     event._value = None
    #     event._ok = True
    #     self.schedule(event)
    # def process(self,event:Event,generator):
    #     event.callbacks = []
    #     event._value = PENDING
    #     event._generator = generator
    #     event._target = self.initialize(self,event)