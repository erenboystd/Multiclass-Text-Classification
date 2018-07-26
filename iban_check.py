
# coding: utf-8


def check_valid_iban(iban_source,ot):

    ot.write(iban_source)
    if len(iban_source) < 26:
        ot.write(iban_source)
        ot.write(" # Unvalid iban \n")
        return 0
    iban_check = iban_source[4:26]
    iban_four = iban_source[0:4]

    firstdigit = iban_four[0]
    secdigit = iban_four[1]

     #changes letters with the values given from CBRT
    firstdigit = ord(firstdigit) - 55
    secdigit = ord(secdigit) - 55
    digits = str(firstdigit)+str(secdigit)

    iban_four = iban_four[2:]
    iban_four = digits+iban_four
    iban_check = iban_check + iban_four

    if int(iban_check)%97 == 1 :
        ot.write(" # Valid iban \n")
        return 1
    else:
        ot.write(" # Unvalid iban \n")
        return 0