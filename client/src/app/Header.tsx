import React from 'react';
import Image from 'next/image';
import { signIn, signOut } from 'next-auth/react';
import githubLogo from './../../public/github-logo.png';

interface HeaderProps {
    session: any;
    selectedLanguage: string;
    setSelectedLanguage: (value: string) => void;
    sendFunction: () => void;
}

const Header: React.FC<HeaderProps> = ({
    session,
    selectedLanguage,
    setSelectedLanguage,
    sendFunction,
}) => {
    return (
        <div className="flex flex-row w-full">
            <div className="bg-slate-900 w-2/3 text-white text-xl py-2 px-4 flex items-center">
                Accelerated Computing Online
            </div>


            {!session?.user?.name ? (

                <div className="flex flex-row w-1/3">
                    <button
                        onClick={() => signIn()}
                        type="button"
                        className="btn btn-primary bg-emerald-500 text-white w-11/12 text-lg"
                    >
                        Sign In
                    </button>
                    <div className="bg-gray-950 w-1/12">
                        <a href="https://github.com/tfsingh/aconline">
                            <Image src={githubLogo} height={53} width={50} alt="github" />
                        </a>
                    </div>
                </div>

            ) : (
                <div className="flex flex-row w-1/3">
                    <select
                        className="bg-blue-500 text-center text-white py-1 px-4 w-4/12 text-base outline-none"
                        onChange={(e) => setSelectedLanguage(e.target.value)}
                        value={selectedLanguage}
                    >
                        <option value="triton">Triton/Numba</option>
                        <option value="cuda">CUDA</option>
                    </select>
                    <button
                        className="bg-emerald-500 text-white w-4/12 py-1 px-4 text-base"
                        onClick={sendFunction}
                    >
                        Run Kernel
                    </button>
                    <button
                        onClick={() => signOut()}
                        type="button"
                        className="bg-red-500 btn btn-primary text-white w-3/12 py-1 px-4 text-base"
                    >
                        Sign Out
                    </button>
                    <div className="bg-gray-950 w-1/12 flex items-center justify-center">
                        <a href="https://github.com/tfsingh/aco">
                            <Image src={githubLogo} height={53} width={50} alt="github" />
                        </a>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Header;
