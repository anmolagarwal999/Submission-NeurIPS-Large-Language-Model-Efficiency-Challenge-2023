```python
def main(**kwargs):
    print("Python env is: ", sys.executable.split('/')[-3])
    print("Script path is: ", __file__)
    print("KWARGS sent to main() are: ", kwargs)
    
    _frac = 1/2
    print("Frac set: ", _frac)
    torch.cuda.set_per_process_memory_fraction(_frac, 0)

```