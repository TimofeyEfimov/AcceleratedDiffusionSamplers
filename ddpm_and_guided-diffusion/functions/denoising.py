# import torch
# import torch.nn.functional as F


# def compute_alpha(beta, t):
#     beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
#     a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
#     return a


# def cond_fn(x, t_discrete, y, classifier, classifier_scale):
#     assert y is not None
#     with torch.enable_grad():
#         x_in = x.detach().requires_grad_(True)
#         logits = classifier(x_in, t_discrete)
#         log_probs = F.log_softmax(logits, dim=-1)
#         selected = log_probs[range(len(logits)), y.view(-1)]
#         return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale


# # def generalized_steps(x, seq, model_fn, b, eta=0, is_cond_classifier=False, classifier=None, classifier_scale=1.0, **model_kwargs):
    
# #     device = x.device
# #     with torch.no_grad():
# #         def model(x, t_discrete):
# #             if is_cond_classifier:
# #                 y = model_kwargs.get("y", None)
# #                 if y is None:
# #                     raise ValueError("For classifier guidance, the label y has to be in the input.")
# #                 noise_uncond = model_fn(x, t_discrete, **model_kwargs)
# #                 cond_grad = cond_fn(x, t_discrete, y, classifier=classifier, classifier_scale=classifier_scale)
# #                 at = compute_alpha(b, t_discrete.long())
# #                 sigma_t = (1 - at).sqrt()
# #                 return noise_uncond - sigma_t * cond_grad
# #             else:
# #                 return model_fn(x, t_discrete, **model_kwargs)
# #         n = x.size(0)
# #         seq_next = [-1] + list(seq[:-1])
# #         x0_preds = []
# #         xs = [x]
        
# #         for i, j in zip(reversed(seq), reversed(seq_next)):
# #             t = (torch.ones(n) * i).to(x.device)
# #             next_t = (torch.ones(n) * j).to(x.device)
# #             at = compute_alpha(b, t.long())
# #             at_next = compute_alpha(b, next_t.long())
# #             xt = xs[-1].to(device)
# #             et = model(xt, t)
# #             x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
# #             x0_preds.append(x0_t.to('cpu'))
# #             c1 = (
# #                 eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
# #             )
# #             c2 = ((1 - at_next) - c1 ** 2).sqrt()
# #             xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
# #             xs.append(xt_next.to('cpu'))

# #     return xs, x0_preds


# def generalized_steps(x, seq, model_fn, b, eta=0, is_cond_classifier=False, classifier=None, classifier_scale=1.0, **model_kwargs):
#     print("inside accelerated")
#     device = x.device
#     # with torch.no_grad():
#     #     def model(x, t_discrete):
#     #         if is_cond_classifier:
#     #             y = model_kwargs.get("y", None)
#     #             if y is None:
#     #                 raise ValueError("For classifier guidance, the label y has to be in the input.")
#     #             noise_uncond = model_fn(x, t_discrete, **model_kwargs)
#     #             cond_grad = cond_fn(x, t_discrete, y, classifier=classifier, classifier_scale=classifier_scale)
#     #             at = compute_alpha(b, t_discrete.long())
#     #             sigma_t = (1 - at).sqrt()
#     #             return noise_uncond - sigma_t * cond_grad
#     #         else:
#     #             return model_fn(x, t_discrete, **model_kwargs)
#     #     n = x.size(0)
#     #     seq_next = [-1] + list(seq[:-1])
#     #     x0_preds = []
#     #     xs = [x]
        
#     #     for i, j in zip(reversed(seq), reversed(seq_next)):
#     #         t = (torch.ones(n) * i).to(x.device)
#     #         next_t = (torch.ones(n) * j).to(x.device)
#     #         at = compute_alpha(b, t.long())
#     #         at_next = compute_alpha(b, next_t.long())
#     #         xt = xs[-1].to(device)
#     #         et = model(xt, t)
#     #         x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
#     #         x0_preds.append(x0_t.to('cpu'))

#     #         newSample = (at_next.sqrt()/at.sqrt()) * xt + ((1-at_next).sqrt()-(at_next.sqrt()/at.sqrt())*((1-at).sqrt()))*et

#     #         #newSample += ((1-at_next).sqrt()-(1-at).sqrt()/(torch.sqrt(at_next / at)))*et
#     #         c1 = (
#     #             eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
#     #         )
#     #         c2 = ((1 - at_next) - c1 ** 2).sqrt()
#     #         xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
#     #         xs.append(newSample.to('cpu'))

#     # Accelerated: 
#     print("the accelerated version")
#     with torch.no_grad():
#         def model(x, t_discrete):
#             if is_cond_classifier:
#                 y = model_kwargs.get("y", None)
#                 if y is None:
#                     raise ValueError("For classifier guidance, the label y has to be in the input.")
#                 noise_uncond = model_fn(x, t_discrete, **model_kwargs)
#                 cond_grad = cond_fn(x, t_discrete, y, classifier=classifier, classifier_scale=classifier_scale)
#                 at = compute_alpha(b, t_discrete.long())
#                 sigma_t = (1 - at).sqrt()
#                 return noise_uncond - sigma_t * cond_grad
#             else:
#                 return model_fn(x, t_discrete, **model_kwargs)
#         n = x.size(0)
#         seq_next = [-1] + list(seq[:-1])
#         x0_preds = []
#         xs = [x]
#         previous_output = None
#         for i, j in zip(reversed(seq), reversed(seq_next)):
#             t = (torch.ones(n) * i).to(x.device)
#             next_t = (torch.ones(n) * j).to(x.device)
#             at = compute_alpha(b, t.long())
#             at_next = compute_alpha(b, next_t.long())
#             xt = xs[-1].to(device)
#             et = model(xt, t)
            
#             x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
#             x0_preds.append(x0_t.to('cpu'))

#             newSample = (at_next.sqrt()/at.sqrt()) * xt + ((1-at_next).sqrt()-(at_next.sqrt()/at.sqrt())*((1-at).sqrt()))*et
#             if previous_output != None: 
#                 term1 = (at_next.sqrt())/(at-at_prev)
#                 term2 = at*((1-at_next).sqrt())/(at_next.sqrt())
#                 term3 = torch.arcsin(at_next.sqrt())
#                 term4 = at*((1-at).sqrt())/(at.sqrt())
#                 term5 = torch.arcsin(at.sqrt())
#                 term6 = (previous_output-et)
#                 newSample += term1*(term2+term3-term4-term5)*term6
#             #newSample += ((1-at_next).sqrt()-(1-at).sqrt()/(torch.sqrt(at_next / at)))*et
#             previous_output = et 
#             at_prev = at 
#             c1 = (
#                 eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
#             )
#             c2 = ((1 - at_next) - c1 ** 2).sqrt()
#             xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
#             xs.append(newSample.to('cpu'))
            
#     return xs, x0_preds

# def ddpm_steps(x, seq, model_fn, b, is_cond_classifier=False, classifier=None, classifier_scale=1.0, **model_kwargs):
    
#     device = x.device
#     with torch.no_grad():
#         def model(x, t_discrete):
#             if is_cond_classifier:
#                 y = model_kwargs.get("y", None)
#                 if y is None:
#                     raise ValueError("For classifier guidance, the label y has to be in the input.")
#                 noise_uncond = model_fn(x, t_discrete, **model_kwargs)
#                 cond_grad = cond_fn(x, t_discrete, y, classifier=classifier, classifier_scale=classifier_scale)
#                 at = compute_alpha(b, t_discrete.long())
#                 sigma_t = (1 - at).sqrt()
#                 return noise_uncond - sigma_t * cond_grad
#             else:
#                 return model_fn(x, t_discrete, **model_kwargs)
#         n = x.size(0)
#         seq_next = [-1] + list(seq[:-1])
#         xs = [x]
#         x0_preds = []
#         betas = b
        
#         for i, j in zip(reversed(seq), reversed(seq_next)):
#             t = (torch.ones(n) * i).to(x.device)
#             next_t = (torch.ones(n) * j).to(x.device)
#             at = compute_alpha(betas, t.long())
#             atm1 = compute_alpha(betas, next_t.long())
#             beta_t = 1 - at / atm1
#             x = xs[-1].to(device)

#             output = model(x, t.float())
#             e = output

#             x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
#             x0_from_e = torch.clamp(x0_from_e, -1, 1)
#             x0_preds.append(x0_from_e.to('cpu'))
#             mean_eps = (
#                 (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
#             ) / (1.0 - at)

#             mean = mean_eps
#             noise = torch.randn_like(x)
#             mask = 1 - (t == 0).float()
#             mask = mask.view(-1, 1, 1, 1)
#             logvar = beta_t.log()
#             sample = mean + mask * torch.exp(0.5 * logvar) * noise
#             xs.append(sample.to('cpu'))
#     return xs, x0_preds

import torch
import torch.nn.functional as F
import math 
import numpy as np

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def cond_fn(x, t_discrete, y, classifier, classifier_scale):
    assert y is not None
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t_discrete)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]
        return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale


def generalized_steps(x, seq, model_fn, b, eta=0, is_cond_classifier=False, classifier=None, classifier_scale=1.0, **model_kwargs):
    print("Normal")
    device = x.device
    # Accelerated: 
    print("the accelerated version")
    with torch.no_grad():
        def model(x, t_discrete):
            if is_cond_classifier:
                y = model_kwargs.get("y", None)
                if y is None:
                    raise ValueError("For classifier guidance, the label y has to be in the input.")
                noise_uncond = model_fn(x, t_discrete, **model_kwargs)
                cond_grad = cond_fn(x, t_discrete, y, classifier=classifier, classifier_scale=classifier_scale)
                at = compute_alpha(b, t_discrete.long())
                sigma_t = (1 - at).sqrt()
                return noise_uncond - sigma_t * cond_grad
            else:
                return model_fn(x, t_discrete, **model_kwargs)
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        previous_output = None
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(device)
            et = model(xt, t)
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            
            newSample = (at_next.sqrt()/at.sqrt()) * xt + ((1-at_next).sqrt()-(at_next.sqrt()/at.sqrt())*((1-at).sqrt()))*et
            # print(torch.sum(torch.isnan(newSample)))
            if previous_output != None: 
                term1 = (at_next.sqrt())/(at-at_prev)
                term2 = at*((1-at_next).sqrt())/(at_next.sqrt())
                term3 = torch.arcsin(at_next.sqrt())
                
                term4 = at*((1-at).sqrt())/(at.sqrt())
                term5 = torch.arcsin(at.sqrt())
                
                term6 = (previous_output-et)
                # print(torch.sum(torch.isnan(term1)), torch.sum(torch.isnan(term6)))
                # print(torch.sum(torch.isnan(term2)), torch.sum(torch.isnan(term3)), torch.sum(torch.isnan(term4)), torch.sum(torch.isnan(term5)))
                # print(torch.sum(torch.isnan(term1*(term2+term3-term4-term5)*term6)))
                #print("Inf checks:", torch.isinf(term1), torch.isinf(term2), torch.isinf(term3), torch.isinf(term4), torch.isinf(term5), torch.isinf(term6))
                full_expression = term1*(term2+term3-term4-term5)*term6
                if not torch.isnan(full_expression).any():  # Check if the full expression has no NaNs
                    newSample += full_expression
                else:
                    print("Skipped updating newSample due to NaN in full expression")
            #previous_output = et 
            at_prev = at 
            c1 = (
                eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(newSample.to('cpu'))
    print("Done")
    return xs, x0_preds

def generalized_steps_accelerated(x, seq, model_fn, b, eta=0, is_cond_classifier=False, classifier=None, classifier_scale=1.0, **model_kwargs):
    print("Normal")
    device = x.device
    # Accelerated: 
    print("the accelerated version")
    with torch.no_grad():
        def model(x, t_discrete):
            if is_cond_classifier:
                y = model_kwargs.get("y", None)
                if y is None:
                    raise ValueError("For classifier guidance, the label y has to be in the input.")
                noise_uncond = model_fn(x, t_discrete, **model_kwargs)
                cond_grad = cond_fn(x, t_discrete, y, classifier=classifier, classifier_scale=classifier_scale)
                at = compute_alpha(b, t_discrete.long())
                sigma_t = (1 - at).sqrt()
                return noise_uncond - sigma_t * cond_grad
            else:
                return model_fn(x, t_discrete, **model_kwargs)
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        previous_output = None
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(device)
            et = model(xt, t)
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            
            newSample = (at_next.sqrt()/at.sqrt()) * xt + ((1-at_next).sqrt()-(at_next.sqrt()/at.sqrt())*((1-at).sqrt()))*et
            # print(torch.sum(torch.isnan(newSample)))
            if previous_output != None: 
                term1 = (at_next.sqrt())/(at-at_prev)
                term2 = at*((1-at_next).sqrt())/(at_next.sqrt())
                term3 = torch.arcsin(at_next.sqrt())
                
                term4 = at*((1-at).sqrt())/(at.sqrt())
                term5 = torch.arcsin(at.sqrt())
                
                term6 = (previous_output-et)
                # print(torch.sum(torch.isnan(term1)), torch.sum(torch.isnan(term6)))
                # print(torch.sum(torch.isnan(term2)), torch.sum(torch.isnan(term3)), torch.sum(torch.isnan(term4)), torch.sum(torch.isnan(term5)))
                # print(torch.sum(torch.isnan(term1*(term2+term3-term4-term5)*term6)))
                #print("Inf checks:", torch.isinf(term1), torch.isinf(term2), torch.isinf(term3), torch.isinf(term4), torch.isinf(term5), torch.isinf(term6))
                full_expression = term1*(term2+term3-term4-term5)*term6
                if not torch.isnan(full_expression).any():  # Check if the full expression has no NaNs
                    newSample += full_expression
                else:
                    print("Skipped updating newSample due to NaN in full expression")
            previous_output = et 
            at_prev = at 
            c1 = (
                eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(newSample.to('cpu'))
    print("Done")
    return xs, x0_preds

def ddpm_steps(x, seq, model_fn, b, is_cond_classifier=False, classifier=None, classifier_scale=1.0, **model_kwargs):
    print("this is accelerated")
    device = x.device
    with torch.no_grad():
        def model(x, t_discrete):
            if is_cond_classifier:
                y = model_kwargs.get("y", None)
                if y is None:
                    raise ValueError("For classifier guidance, the label y has to be in the input.")
                noise_uncond = model_fn(x, t_discrete, **model_kwargs)
                cond_grad = cond_fn(x, t_discrete, y, classifier=classifier, classifier_scale=classifier_scale)
                at = compute_alpha(b, t_discrete.long())
                sigma_t = (1 - at).sqrt()
                return noise_uncond - sigma_t * cond_grad
            else:
                return model_fn(x, t_discrete, **model_kwargs)
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to(device)
            
            output = model(x, t.float())
            output = 0
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)
            
            mean = mean_eps
            noise = torch.randn_like(x)
            noise2 = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            #sample = mean + mask * torch.exp(0.5 * logvar) * noise
            sample = mean + mask * beta_t.sqrt() * noise
            
            # Vanilla 
            sample = 1/((at / atm1).sqrt())*(x- (1-at / atm1) * (1/(1-at).sqrt())* e)+ mask * beta_t.sqrt() * noise
            # (at/atm1).sqrt()*
            # term1 = x+((at/atm1).sqrt())*(((1-at / atm1)/2).sqrt())*noise2 
            # newScore = model(term1, t.float())
            # sample = 1/((at / atm1).sqrt())*(term1- (1-at / atm1) * (1/(1-at).sqrt())* newScore)+ mask * (beta_t/2).sqrt() * noise

            xs.append(sample.to('cpu'))
    return xs, x0_preds

def ddpm_steps_accelerated(x, seq, model_fn, b, is_cond_classifier=False, classifier=None, classifier_scale=1.0, **model_kwargs):
    print("this is accelerated")
    device = x.device
    with torch.no_grad():
        def model(x, t_discrete):
            if is_cond_classifier:
                y = model_kwargs.get("y", None)
                if y is None:
                    raise ValueError("For classifier guidance, the label y has to be in the input.")
                noise_uncond = model_fn(x, t_discrete, **model_kwargs)
                cond_grad = cond_fn(x, t_discrete, y, classifier=classifier, classifier_scale=classifier_scale)
                at = compute_alpha(b, t_discrete.long())
                sigma_t = (1 - at).sqrt()
                return noise_uncond - sigma_t * cond_grad
            else:
                return model_fn(x, t_discrete, **model_kwargs)
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to(device)
            
            #output = model(x, t.float())
            output = 0
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)
            
            mean = mean_eps
            noise = torch.randn_like(x)
            noise2 = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            #sample = mean + mask * torch.exp(0.5 * logvar) * noise
            sample = mean + mask * beta_t.sqrt() * noise
            
            # Vanilla 
            #sample = 1/((at / atm1).sqrt())*(x- (1-at / atm1) * (1/(1-at).sqrt())* e)+ mask * beta_t.sqrt() * noise
            # (at/atm1).sqrt()*
            term1 = x+((at/atm1).sqrt())*(((1-at / atm1)/2).sqrt())*noise2 
            newScore = model(term1, t.float())
            sample = 1/((at / atm1).sqrt())*(term1- (1-at / atm1) * (1/(1-at).sqrt())* newScore)+ mask * (beta_t/2).sqrt() * noise

            xs.append(sample.to('cpu'))
    return xs, x0_preds


# def ddpm_steps_accelerated(x, seq, model_fn, b, is_cond_classifier=False, classifier=None, classifier_scale=1.0, **model_kwargs):
#     device = x.device
#     with torch.no_grad():
#         def model(x, t_discrete):
#             if is_cond_classifier:
#                 y = model_kwargs.get("y", None)
#                 if y is None:
#                     raise ValueError("For classifier guidance, the label y has to be in the input.")
#                 noise_uncond = model_fn(x, t_discrete, **model_kwargs)
#                 cond_grad = cond_fn(x, t_discrete, y, classifier=classifier, classifier_scale=classifier_scale)
#                 at = compute_alpha(b, t_discrete.long())
#                 sigma_t = (1 - at).sqrt()
#                 return noise_uncond - sigma_t * cond_grad
#             else:
#                 return model_fn(x, t_discrete, **model_kwargs)
#         n = x.size(0)
#         seq_next = [-1] + list(seq[:-1])
#         xs = [x]
#         x0_preds = []
#         betas = b
#         for i, j in zip(reversed(seq), reversed(seq_next)):
#             t = (torch.ones(n) * i).to(x.device)
#             next_t = (torch.ones(n) * j).to(x.device)
#             at = compute_alpha(betas, t.long())
#             atm1 = compute_alpha(betas, next_t.long())
#             beta_t = 1 - at / atm1
#             x = xs[-1].to(device)

#             output = model(x, t.float())
#             e = output

#             x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
#             x0_from_e = torch.clamp(x0_from_e, -1, 1)
#             x0_preds.append(x0_from_e.to('cpu'))
#             mean_eps = (
#                 (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
#             ) / (1.0 - at)

#             mean = mean_eps
#             noise = torch.randn_like(x)
#             mask = 1 - (t == 0).float()
#             mask = mask.view(-1, 1, 1, 1)
#             logvar = beta_t.log()
#             #sample = mean + mask * torch.exp(0.5 * logvar) * noise
#             sample = mean + mask * beta_t * torch.exp(0.5) * noise

#             xs.append(sample.to('cpu'))
#     return xs, x0_preds


# import torch
# import torch.nn.functional as F


# def compute_alpha(beta, t):
#     beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
#     a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
#     return a


# def cond_fn(x, t_discrete, y, classifier, classifier_scale):
#     assert y is not None
#     with torch.enable_grad():
#         x_in = x.detach().requires_grad_(True)
#         logits = classifier(x_in, t_discrete)
#         log_probs = F.log_softmax(logits, dim=-1)
#         selected = log_probs[range(len(logits)), y.view(-1)]
#         return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale


# def generalized_steps(x, seq, model_fn, b, eta=0, is_cond_classifier=False, classifier=None, classifier_scale=1.0, **model_kwargs):
#     device = x.device
#     with torch.no_grad():
#         def model(x, t_discrete):
#             if is_cond_classifier:
#                 y = model_kwargs.get("y", None)
#                 if y is None:
#                     raise ValueError("For classifier guidance, the label y has to be in the input.")
#                 noise_uncond = model_fn(x, t_discrete, **model_kwargs)
#                 cond_grad = cond_fn(x, t_discrete, y, classifier=classifier, classifier_scale=classifier_scale)
#                 at = compute_alpha(b, t_discrete.long())
#                 sigma_t = (1 - at).sqrt()
#                 return noise_uncond - sigma_t * cond_grad
#             else:
#                 return model_fn(x, t_discrete, **model_kwargs)
#         n = x.size(0)
#         seq_next = [-1] + list(seq[:-1])
#         x0_preds = []
#         xs = [x]
#         for i, j in zip(reversed(seq), reversed(seq_next)):
#             t = (torch.ones(n) * i).to(x.device)
#             next_t = (torch.ones(n) * j).to(x.device)
#             at = compute_alpha(b, t.long())
#             at_next = compute_alpha(b, next_t.long())
#             xt = xs[-1].to(device)
#             et = model(xt, t)
#             x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
#             x0_preds.append(x0_t.to('cpu'))
#             c1 = (
#                 eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
#             )
#             c2 = ((1 - at_next) - c1 ** 2).sqrt()
#             xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
#             xs.append(xt_next.to('cpu'))

#     return xs, x0_preds


# def ddpm_steps(x, seq, model_fn, b, is_cond_classifier=False, classifier=None, classifier_scale=1.0, **model_kwargs):
#     device = x.device
#     with torch.no_grad():
#         def model(x, t_discrete):
#             if is_cond_classifier:
#                 y = model_kwargs.get("y", None)
#                 if y is None:
#                     raise ValueError("For classifier guidance, the label y has to be in the input.")
#                 noise_uncond = model_fn(x, t_discrete, **model_kwargs)
#                 cond_grad = cond_fn(x, t_discrete, y, classifier=classifier, classifier_scale=classifier_scale)
#                 at = compute_alpha(b, t_discrete.long())
#                 sigma_t = (1 - at).sqrt()
#                 return noise_uncond - sigma_t * cond_grad
#             else:
#                 return model_fn(x, t_discrete, **model_kwargs)
#         n = x.size(0)
#         seq_next = [-1] + list(seq[:-1])
#         xs = [x]
#         x0_preds = []
#         betas = b
#         for i, j in zip(reversed(seq), reversed(seq_next)):
#             t = (torch.ones(n) * i).to(x.device)
#             next_t = (torch.ones(n) * j).to(x.device)
#             at = compute_alpha(betas, t.long())
#             atm1 = compute_alpha(betas, next_t.long())
#             beta_t = 1 - at / atm1
#             x = xs[-1].to(device)

#             output = model(x, t.float())
#             e = output

#             x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
#             x0_from_e = torch.clamp(x0_from_e, -1, 1)
#             x0_preds.append(x0_from_e.to('cpu'))
#             mean_eps = (
#                 (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
#             ) / (1.0 - at)

#             mean = mean_eps
#             noise = torch.randn_like(x)
#             mask = 1 - (t == 0).float()
#             mask = mask.view(-1, 1, 1, 1)
#             logvar = beta_t.log()
#             sample = mean + mask * torch.exp(0.5 * logvar) * noise
#             xs.append(sample.to('cpu'))
#     return xs, x0_preds