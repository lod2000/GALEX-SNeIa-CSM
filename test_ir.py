from galex_injection_recovery import Injection

inj = Injection.from_name('SN2007on', 'NUV', 700, 1)
recovered = inj.recover(3)
print(recovered)