def squash(s): # 기본
        s_norm = torch.norm(s, p=2, dim=-1, keepdim=True)
        s_norm2 = torch.pow(s_norm, 2)
        v = (s_norm2 / (1.0 + s_norm2)) * (s / s_norm)
        return v

def squash(s): #sq by L1
        #s_norm = torch.norm(s, p=2, dim=-1, keepdim=True)
        L1_norm = (torch.abs(s)).sum(-1, keepdim=True)
        #s_norm2 = torch.pow(s_norm, 2)
        v = L1_norm * s / (1 + (L1_norm ** 2))
        #v = (s_norm2 / (1.0 + s_norm2)) * (s / s_norm)
        return v

def squash(s): #unit scaling by L1
        #s_norm = torch.norm(s, p=2, dim=-1, keepdim=True)
        L1_norm = (torch.abs(s)).sum(-1, keepdim=True)
        #s_norm2 = torch.pow(s_norm, 2)
        v = s / L1_norm
        #v = (s_norm2 / (1.0 + s_norm2)) * (s / s_norm)
        return v

def squash(s): #unit scaling approx by L1
        #s_norm = torch.norm(s, p=2, dim=-1, keepdim=True)
        L1_norm = (torch.abs(s)).sum(-1, keepdim=True)
        L1_norm = torch.round(torch.log2(torch.ceil(1 + L1_norm)))
        L1_norm_bit_shift = torch.pow(2, L1_norm)
        #s_norm2 = torch.pow(s_norm, 2)
        v = s / L1_norm_bit_shift
        #v = (s_norm2 / (1.0 + s_norm2)) * (s / s_norm)
        return v
