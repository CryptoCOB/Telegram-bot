#!/usr/bin/env python3
import csv

# In practice, you might parse the raw chat text to populate this list dynamically,
# but for now, we'll hardcode the user-data pairs as provided.
solana_data = [
    ("Timotius Vincent", "8pGJynk7khku8kU74KXftRYXkh7HEaMqntiapzHqnYGN"),
    ("Kenzii. ❤️TravelFrog🐸 🌱SEED| Drops💧", "8bjNYY7FMtiGCvAPbnsUsjTGMyv2HYgstj7YbRTkLuA9"),
    ("PakLalet | Drops💧", "DN1FfMCS1aWrVu6pE8tspSvDWAdgsPpfsS5iYemBGwFN"),
    ("Babi Hutan🍅🐾", "7YiLR5ExtbtGVokcKWP5epTgVbWWqtwgMgmdzQJrdyxS"),
    ("🐤 🐾▪️ 🐵 🐰🍋🌱SEED🐈‍⬛💠 $VROG", "HacJRe6NLGVm9gutCmcTLZrzFHgRUYWCQHirAPbJqiCk"),
    ("Ayangkayess ▪️", "3EJvqTx1eyGCMDGpBzyMEPnjC7VTqKC8m6BPQvZrWkUD"),
    ("Lukman Triono 🍅▪️🐾🐐", "5Rhy73VguSr1HC91z5aFXspcBjt9knYX4gko2p8JsoXW"),
    ("Ahmad Fahri 🍅 ▪️🐾| Drops💧 🌱SEED", "EzZfd6Dss2uMRCYEdH32XnfvypHZwHq4XBeTDFw4h9WM"),
    ("Aii Botutihe 🌱SEED | 🐾", "9hpGccEc62WWgewMuBfn5KjbbKHbjYrTdPNAbEdhpTGZ"),
    ("Victorezo", "7rmBAWwoFnv5aLhpqCf4UkGaMnVLwwB41BGT8YLBGnBW"),
    ("Tes 🍅 Tes 🐈‍⬛ ▪️ 🐾| Drops💧", "6L33v4GCnbmufvS1JboZgeaK6W7FFPr3bjhfXqZDETsc"),
    ("Bullszcrypto | Drops", "E8HcxS5nKc7FWSM65jTBvRhSXPtdCqrJAvvpddFqrH8P"),
    ("Lukman Triono 🍅▪️🐾🐐 (again)", "5Rhy73VguSr1HC91z5aFXspcBjt9knYX4gko2p8JsoXW"),
    ("Marrr Loong 🐾🌱SEED", "7Fck8UdQfpSNWyXS9xhYy1THeHFA6Baca4QEpTx2Lg3R"),
    ("SQ One🐾", "85QfD1E7z3WjnWuvZEtAUoBsFktRYTYC68k1QxL3hxbu"),
    ("Angel", "A83Jxf6wLPQ5VkfJMBMUY2S9PYBNL9KVdpQzE5wnSgXB"),
    ("ayun pasa", "BZYBnkkaDtTdzWX8nrSTKucexjk5AJQyusmYnLfNGPhD"),
    ("Nur Sugianto 🌱SEED 🐦SUI 🐥🐾🍋🐐▪️| Drops💧🐸 🍒🐱CATGOLD $VROG Oxyz", "9pp34Udv6svrZL6b8d3vNuN2gZ5sbCbdMaxLBgz3PUyM"),
    ("MYSTERY | Drops💧 IDOSOL.me", "H7BP2VRssDykt4Qg68e3FQpDBffcX66TasSZmuic6kGN"),
    ("🐤 sigit 🚀PoPP 🐥 Gra-Gra 🌱SEED 🐦 SUI 🐾▪️", "3qM3RPaSrBjtbWpRPFPD3Thxse5LgKuJMKiQSp6ciHQf"),
    ("jumbaliali", "6uLayS3spCvv8cem3V7f7dui8gqcosZauHdUuo2imGck"),
    ("Rhnn 🌱SEED", "BugvokViHd1Qtci7KYrBErNTjjg3otyKn2NhciwFJuQv"),
    ("Yummy | 🏠 ClickCity 🍅▪️🐾🐐🐦📦 Bums| Drops💧🌱SEED🐾 BabyDoge PAWS", "88vMyPXbzjiWQgsg7Dbmcg2WSF638sgRcdXGm46EAVmX"),
    ("bsk ncy▪️🐾", "5GzyKzFKgkfjQx3PfVMs5ntqs5eCL6STQV1W8X8jbXrr"),
    ("Afsheen Qiana 🐾 🍅 | Drops💧$VFROG", "J6bqXa99aamBjzVuoi1EHC5gqcSE4w88aNiJH5qy6LGc"),
    ("yandi", "DXERBWebHGNquvJG31MmtyHmnm5douR5xkfbgSQPXaFp"),
    ("Kif🦆 FLIES 🪰🥠", "BJoYFbH22rJv4MzeE4cyF4xJmbrnY9hXeJRYjxKwbs3"),
    ("Onlinecoin 🐤 Gra-Gra 🦴🌱SEED WAVE 🌊 ▪️🐾 | Drops💧", "2HSQfYvAnVsEBFeHRYYxYoSybNU7NqUV3RJ835tMsVwD"),
    ("ᶦᶰᵈKingStore▪️🐾", "3BHtuuLDoWsYtcrJEQA1U84Ma8TStXWjnSKH5cFDqtap"),
    ("Adi▪️ .Oxyz", "3f9yxeoPopU8HfzNXc4itEXco94VVpp1cenBWACRLGSJ"),
    ("Zaa 🦴", "FfqhkYyhQrnBF4Jq7zHBMmhNTG2x9QdksDcxXHnGA693"),
    ("Eki Pratama▪️🐾 🌱SEED | Drops💧", "GCFJAST44PrN1whj2bGSB7y9HCQNWgNv7pVN1p4SxvxY"),
    (".🐾", "UFHDrSsvfX2KGZSHhtSsZMwnP9nbxFZBbJxn6K4Mgbd"),
    ("Gimlyyz (sweet revenge/acc) | Drops💧", "GWpNGX6Pc8cCjDUatVrUnyrrtw3ZdKwUhoEUehkPQ5ZL"),
    ("Sandra🍅🐾 Amelia | Drops💧", "Ea4ptB3hP9hGzpEWAsrG8pWbBVDrLUJXB14qjsTYWNt3"),
    ("Raya X", "BHWTzEP3QcCemQQeA8xHSMvi5HdKNJLCS9KXypnPwadM"),
    ("aldi Siag", "5DvP8Ha8fAXMvvZnaAeCQ28e93nW9pHsbZVYSjVH47mk"),
    # Heromine L 🐾 $VROG had no address below them, so we skip or store blank:
    # ("Heromine L 🐾 $VROG", ""),

    ("Lukman Triono 🍅▪️🐾🐐 (duplicate)", "85QfD1E7z3WjnWuvZEtAUoBsFktRYTYC68k1QxL3hxbu"),
    ("Mahnod 🌱SEED 🍅 🐾 Ⓜ️ MEMES", "GCHNwHAcnWFjUwQRsWv11Yh4HQhsKmvodFcex4g7JpYh"),
    ("Veyz™🐾| Drops💧 ▪️🌱SEED", "9pLdFF9bqmy7r7n6pHT54aH2zvpbFzFPwQvvu1FV6qai"),
    ("SUPER GEMBULLL", "3AxMHqDKWB3wUJjgV6mGqfDVwQqdQ1c8ShZSYnsPyeM3"),
    (". 🌱SEED 🐾 BabyDoge PAWS || INSYALLAH JP💠 🍅▪️🐾 🍋 | Drops💧", "szrXJ1CMaiZ9LSbxCky85MQzbCqpK3TKdqxRB61UTWH"),
    ("REZZA(❤️TravelFrog🐸)...", "8njWxPKGXhdHGAQPYYJW7EBk9GZTdtG1WddZHrLKmG2Q"),
    ("You call me bek 🥔", "EXtgJk4YpZTSMhX4PpPdFGMHFn432BA9unjzKeomQRmY"),
    ("ᴊᴀᴄᴋ | SANG SANNIN 🐦 SUI 🐾 🌱SEED | Drops💧", "CWnmhzCwMipJwCa88xp6SmGtarhiZeEf7cC9qMHBp2hP"),
    ("Miscella $VROG Drops💧🌱SEED 🐾 ▪️", "8pJrbY5Ex4wSMpM36uuZLs59UjeuauqP1Zb4NsjD3qxd"),
    ("Kojojago02™🐾🌱SEED ▪️", "1HL1zCPe2Jm3MPwVJh3nD1o3zDb8HwbVqfQzgbmpRft"),
    ("JODS | Drops💧 🌱SEED", "5xHy2nWzQ9EASHPCNsvjDtm7GYEgxovEjNLAttF41KMj"),
    ("Zyaa | Cryptokom.Oxyz", "Fr6EzawMqUgGohDNdozmWWHczqHQhUdzuf27eJcosXWn"),
    ("optx208 🌱SEED", "BzVg87euocCbaVjRwmXhfkmAaRpEDGGjLmiR7s34icmW"),
    ("Rizky Dani 🐾| Drops💧", "A8dMHadFgLaeRbyWmoKCetiiReFBTcV4AhuPCatcj1PX"),
    ("Papi Chulo 🐾", "2DCMVtpkV8r7dP2RaLrGxDdkCU58vGFod2s7Bv6GEPzA"),
    ("demonshunt", "BLJDwP4rdKrFdcW7UtUB97cCgeKkTy82S8ix9sQEuKt1"),
    ("jumbaliali (again)", "6uLayS3spCvv8cem3V7f7dui8gqcosZauHdUuo2imGck"),
    ("Nurkholik | Drops💧🍅", "BS1oWRQF58CbmhqnNSgSMdxs8rzhoLaejshqXD5A1hBo"),
    ("VennXD 🐾| Drops💧", "JBwP8j3DhxmJh97hB6KUWdyYuRw8uHoYGnhefxxjMB3g"),
    ("Rece Shang🌱SEED🍅🍋🧑🏻‍💻", "63aZJQM2BjWNNjM8bLGwLb9dacNCSDuoyPGMZg1c2g8t"),
    ("𝙏𝙖𝙣𝙖𝙯𝙪 𝕏▪️| Drops💧", "FBrgk1GTPQuHbSNhCjaGFD73KUmGBFigA3kHBBsa1k6z"),
    ("Rizky Dani 🐾| Drops💧 (again)", "DfHUJrfEHaYg7Y4nBPjxQQSYL5AFzZthsnygsWizj2vF"),
    ("Angel (2)", "B5fWxVAhNqSd89KKfv8U1y6LJ5tSUbXCNWYwHrgEJkUL"),
    ("alfin▪️🐾 ramadan", "6b8yWKrMXwbmah3fmuZXDyw7zKwAZQp3dDHi2HESC4qR"),
    ("obengasep581▪️🐾", "8rYgSZsASHuUC2FqovwjtBQ4T2u1mDRzBwBejzcctHMB"),
    # If a user posted a raw address with no preceding username, we might store an empty user or skip it:
    ("(No username)", "GLqpWKrFp1c1uE615wWru76z3cmgPvigjeRStc22rr4P"),
    ("WARR🐾| Drops💧$SOLCLAIM Roomm", "HatucnCy3j6dyC4RA5HbMP792DgSvxzku9oGhYqfJaD8"),
    ("Ari 🐾", "42TdXD1XpocMPq11RpM3owwJ2QPzsDQkvwmxBoJjMRfy"),
    ("pastijp🦴| Drops💧🐸| $EMOJI", "CPMu1p6aHudH1J42Z3gM3dHWw6o9GQBMf1d4UjAwZaoS"),
    ("markonah🌱SEED | Drops💧", "AUFCPYwC5m61MZZovafD2Na89EpYq33tqpFjfAYVrDuf"),
    ("hantu caruluk| Drops💧🌱SEED", "GWKvXz9MFXaPTW3Ay2AcqKbR6xtqKUADriXQcuptDyKu"),
    # If no user is found for an address:
    ("(No username)", "3AfJz3TBsSr8ycdMwvCaaSSCZmAuMLAHSaLdXnu8QvhM"),
    ("fallfor🐾", "AFWLr851r6JZBEECsG3aTH4cofk2XRwu2zVourtm8aEA"),
    ("Hakku🐾", "EM4JPMsYuBpYZkULfbFZrtTcZ1XcBE686ZXpfCuiBauk"),
    ("10taclesss🐾", "DNy4oeSmNazpU62wKtLDRjTEbsZ4z13xot7K8Hq8fTNR"),
    ("Hery Setiawan", "AcFkrvt1VuX5d4iLMaS2rWkXNockPiDHe9vdV9AqwgFP"),
    ("surya15🐾", "7qMDFZmmhV39ZQohPqNyCLxyQzNqXUDANPQutfQeTKUh"),
    ("yokosooh🐾", "95BXTr45VrTYpdscivvyoA5bShaZd5m6gr2wFSN6jBFk"),
    ("faiyahhhh🐾", "DgYpZAgvnvR7hBJo8SpXu3uicD3nTBz9nUtv68BGRrgH"),
    ("Deny Indriyana", "9jaLmGvTrd5KfdpnHZFJi1aAwYq2MUMZEfEnZNB6rZBr"),
    ("weniweb3🐾", "GmBQJs99rHuAabcm8mhm6bueswXsrsGXWUnMYVqE6nfF"),
    ("onedepok| Drops💧DUCKS 🦆 $SOLCLAIM Warr", "6kDzH3UCkmNDBzjhh9HVomSehreY3grqdm8YgUQWPaT2"),
    ("kiyy ▪️", "cpjAqq1gFmf1Zc244gpP5E3attas2FJdNpWAoFfGFHU"),
    # Dores Marlins had no address:
    ("Dores Marlins | Drops💧 🌱SEED 🐾▪️", ""),
]


def create_solana_csv(filename: str = "solana_addresses.csv"):
    """
    Saves the hardcoded username–address data to a CSV file.
    """
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Username", "Solana_Address"])  # header row
        for username, address in solana_data:
            # Only write rows that have at least an address
            # (Or remove the 'if address' check if you want blank addresses included)
            if address:
                writer.writerow([username, address])

    print(f"CSV file '{filename}' created successfully!")


if __name__ == "__main__":
    create_solana_csv()