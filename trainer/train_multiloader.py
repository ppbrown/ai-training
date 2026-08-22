from typing import Iterable, List, Any


class InfiniteLoader:
    """ 
    Expectation is to init with one or more DataLoaders.
    Caller can then infinitely loop around the data,
    round-robin one batch per dataloader.
    Note that each dataloader is expected to have  drop_last=True set,
    or you will get corrupted training.
    """
    def __init__(self, *loaders: Iterable):
        if not loaders:
            cls_name = type(self).__name__
            raise ValueError(f"{cls_name} requires at least one loader.")
        self.loaders = list(loaders)
        self._iters = [iter(ld) for ld in self.loaders]

        self._shortest_len = min(len(ld) for ld in self.loaders)

    def __iter__(self):
        return self._generator()

    def _generator(self):
        idx = 0
        n = len(self.loaders)
        step = 0

        while True:
            ld = self.loaders[idx]
            it = self._iters[idx]

            try:
                batch = next(it)
            except StopIteration:
                it = iter(ld)
                self._iters[idx] = it
                batch = next(it)

            yield step, batch
            step += 1

            idx = (idx + 1) % n


    def get_shortest_len(self) -> int:
        """
        Return the minimum len() of the underlying loaders.
        Useful for epoch lenth calculationns
        """
        return self._shortest_len

    def shutdown(self):
        """
        Explicitly tear down each underlying DataLoader's worker
        processes now, instead of leaving it to GC/atexit timing -
        which can leave persistent_workers running for a noticeable
        stretch after the script has already returned control to the
        shell.
        """
        for it in self._iters:
            shutdown_fn = getattr(it, "_shutdown_workers", None)
            if shutdown_fn is None:
                continue
            try:
                shutdown_fn()
            except Exception as e:
                print("warning: error shutting down dataloader workers:", e)
        self._iters = []
