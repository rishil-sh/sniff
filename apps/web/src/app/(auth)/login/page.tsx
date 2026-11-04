"use client";

import { useState } from "react";

import SignInForm from "@/components/sign-in-form";
import SignUpForm from "@/components/sign-up-form";

const LoginPage = () => {
  const [showSignIn, setShowSignIn] = useState(true);

  return showSignIn ? (
    <SignInForm onSwitchToSignUp={() => setShowSignIn(false)} />
  ) : (
    <SignUpForm onSwitchToSignIn={() => setShowSignIn(true)} />
  );
};

export default LoginPage;
