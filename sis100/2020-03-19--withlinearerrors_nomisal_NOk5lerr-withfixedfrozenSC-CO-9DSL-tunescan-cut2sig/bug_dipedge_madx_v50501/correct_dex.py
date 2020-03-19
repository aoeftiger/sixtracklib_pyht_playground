from cpymad import madx

m = madx.Madx()
m.echo = False
m.info = False
m.warn = False

m.input('''
a: sbend, l = 0.1, angle = 0.01, e1 = 0.005, hgap = 0.034, fint = 0.35;

fodo: sequence, l = 10;
a, at = 5;
endsequence;
''')

m.command.beam(particle='proton', energy='1')

m.use('fodo')

m.command.select(flag='makethin', class_='sbend', slice_='9')
m.command.makethin(makedipedge=True, style='teapot', sequence='fodo')

print ('\n\n\n' + '*'*80 + '\n')
print (m.sequence.fodo.elements[-2])
print ('  ==> fint is now finite and identical to leading dipedge.fint!')
print ('\n' + '*'*80 + '\n\n\n')
