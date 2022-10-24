using PyCall
time=PyCall.pyimport("time")
start=time.time()
using LinearAlgebra,Statistics

pyscf=PyCall.pyimport("pyscf")
function make_molecule()
    mol=pyscf.gto.M()
    atoms="O 0.000000000000 -0.143225816552 0.000000000000;H 1.638036840407 1.136548822547 -0.000000000000;H -1.638036840407 1.136548822547 -0.000000000000"
    mol.charge=0
    mol.unit = "Bohr"
    mol.spin=0
    mol.build(
	    atom= atoms,
	    basis = "sto3g")
	    #basis = "cc-pVDZ")
    return mol
end
mol=make_molecule()
#print(mol.atom)
#print(mol.basis)
function pyscf_1e()
    h1e = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    return h1e
end

function pyscf_overlap()
    S = mol.intor("int1e_ovlp")
    return S
end

function pyscf_2e()
    eri = mol.intor("int2e")
    return eri
end

function pyscf_nucr()
    nuclear_repulsion = mol.energy_nuc()
    constant = nuclear_repulsion
    return constant
end
n_elec=10
no=Int64((n_elec)/2)
function scf_energy(D,fock,h1e)
    h=[]
    h,nbasis=size(D)
    n_hf_energy=0.0
    for i in 1:nbasis
        for j in 1:nbasis
            n_hf_energy+=(D[i,j]*(fock[i,j]+h1e[i,j]))
        end
    end
    return n_hf_energy
end

function make_fock(D,h1e,eri)
    h=[]
    h,nbasis=size(D)
    fock= zeros(Float64,nbasis,nbasis)
    for i in 1:nbasis
        for j in 1:nbasis
            for k in 1:nbasis
                for l in 1:nbasis
                    fock[i,j]+=(D[k,l]*((2*eri[i,j,k,l])-eri[i,k,j,l]))
                #push!(new_fock[i,j],(*(D[k,l],(*((2*twoe[i,j,k,l])-twoe[i,l,k,j])))))
                end
            end
        end
    end 
    return fock+h1e
end
function make_density(c,no)
    h=[]
    h,nbasis=size(c)
    D=zeros(Float64,nbasis,nbasis)
    for i in 1:nbasis
        for j in 1:nbasis
            for m in 1:no
                D[i,j]+=(c[i,m]*c[j,m])
                #push!(D,(*(c[i,m],c[j,m])))
            end
        end
    end   
    return D
end
h1e=pyscf_1e()
eri=pyscf_2e()
constant=pyscf_nucr()
S=pyscf_overlap()

#println("the one electron integral is ",h1e ,"\n")
#println("the two electron integral is ",eri,"\n")
#println(" shape of two electron integral is",size(eri),"\n")
#println("the nuclear nuclear repulsion term is ",constant,"\n")
#println("the overlap integral is",S)
function make_s_half() 
    s=(S+S')/2
    #println(nbasis)
    q,L=LinearAlgebra.eigen(s)
    q_half=[]
    for i in 1:lastindex(q)
        push!(q_half,q[i]^(-0.5))
    end
    q_half=Diagonal(q_half)
    s_half=(L*q_half)
    s_half=(s_half*L')
    #println("the value of s_half is    ",s_half)
    return s_half
end 
#scf initialisation 

function scf()
    s=(S+transpose(S))/2
    h,nbasis=size(s)
    fock= zeros(Float64,nbasis,nbasis)
    init_fock= (h1e*make_s_half()')
    init_fock=(make_s_half()*init_fock)
    init_fock=(init_fock+init_fock')/2
    Hartree_fock_energy=0.0
    E=[]
    del_e=0.0
    e,c0 = LinearAlgebra.eigen(init_fock)
    c = (make_s_half()*c0)
    list_e=[0.0,]
    fock= zeros(Float64,nbasis,nbasis)
    for n in 1:100
        if n==1
            D=zeros(Float64,nbasis,nbasis)
            for i in 1:no
                D[i,i]+=1.0
            end
        end
        D=make_density(c,no)
        fock=make_fock(D,h1e,eri)
        f_dash= (fock*make_s_half()')
        f_dash=(make_s_half()*f_dash)
        f_dash=(f_dash+transpose(f_dash))/2
        eps,c_dash=LinearAlgebra.eigen(f_dash)
        E=eps
        c=(make_s_half()*c_dash)
        D=make_density(c,no)
        hf_energy=scf_energy(D,fock,h1e)
        push!(list_e,sum(hf_energy)) 
        #println(list_e)
        for i in list_e
            del_e=list_e[lastindex(list_e)]-list_e[lastindex(list_e)-1]
        end
        #println(del_e)
        if (abs(del_e))<=10^(-12)
            break
        end
        Hartree_fock_energy=sum(hf_energy)+constant
        println("iteration= ",n,"    energy= ", sum(hf_energy)+constant,"         delta_e= ",round(del_e,digits=12))
    end
    return Hartree_fock_energy,E,c,fock,nbasis
end
Hartree_fock_energy,E,c,fock,nbasis=scf()
println("final Hartree fock energy= ", Hartree_fock_energy)
#println(E)
println("the value of coefficient matrix is",c)
#println("the size of fock is ",size(fock))
endtime=time.time()
println("the runtime of the code is", endtime-start)
